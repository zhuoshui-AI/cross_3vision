"""RGB + Infrared + Depth multimodal detection dataset and augmentations.

Directory layout under data_root (per spec):
    visible/<stem>.{jpg|png}   3-channel 8-bit visible (RGB)
    infrared/<stem>.{jpg|png}  3-channel 8-bit thermal (stacked single channel)
    depth/<stem>.png           1-channel 16-bit mm
    labels/<stem>.txt          [cls, cx, cy, w, h] normalized
    splits/<split>.txt         one stem per line (optional)
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations import BboxParams

# Albumentations changed geometric-transform signatures across 2.x releases:
# RandomResizedCrop takes `size=(h, w)`, while Resize kept (or later regained)
# `height`/`width` depending on the exact version. Rather than guessing from
# the version number, introspect each transform's own InitSchema and pass the
# keyword it actually supports.
def _schema_fields(cls):
    """Return the set of size-related fields declared by an A. transform."""
    schema = getattr(cls, "InitSchema", None)
    if schema is None:
        return set()
    fields = getattr(schema, "model_fields", None)
    if fields is not None:          # pydantic v2
        return set(fields.keys())
    return set(getattr(schema, "__fields__", {}).keys())  # pydantic v1


def _make_transform(cls, size, **extra):
    """Construct a geometric transform with the size kwargs it supports."""
    fields = _schema_fields(cls)
    if "size" in fields:
        return cls(size=(size, size), **extra)
    return cls(height=size, width=size, **extra)

IMG_EXTS = (".jpg", ".jpeg", ".png")

# ImageNet statistics for RGB branch (Swin expects these).
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_image(stem, directory):
    """Locate an image by stem under directory, tolerating .jpg/.png mix."""
    for ext in IMG_EXTS:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No image for stem '{stem}' in {directory}")


def load_label(txt_path, num_classes):
    """Parse [cls, cx, cy, w, h] (normalized) → xyxy normalized + class ids.

    Returns (boxes_xyxy_norm [N,4] float32, labels [N] int64).
    """
    boxes, labels = [], []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls < 0 or cls >= num_classes:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                x2, y2 = cx + w / 2.0, cy + h / 2.0
                # Boxes hugging the image edge can land a hair outside [0,1]
                # from float rounding (e.g. -2e-7); albumentations 2.x rejects
                # these in its strict range check.
                x1 = min(max(x1, 0.0), 1.0)
                y1 = min(max(y1, 0.0), 1.0)
                x2 = min(max(x2, 0.0), 1.0)
                y2 = min(max(y2, 0.0), 1.0)
                # Drop degenerate boxes (zero/negative area after clipping).
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64))


class MultiModalDataset(Dataset):
    """Aligned RGB / IR / Depth dataset for DETR-style training.

    Each sample returns a dict:
        pixel_values_rgb   (3, H, W) float32, ImageNet-normalized
        pixel_values_ir    (1, H, W) float32, /255
        pixel_values_depth (1, H, W) float32, /20000 with 0/<10mm masked
        depth_mask         (H, W) bool, True where depth is valid
        pixel_mask         (H, W) bool, all True (padding handled elsewhere)
        labels             dict(class_labels [N], boxes [N,4] cxcywh normalized — matches DETR matcher/loss)
        image_id           long
        stem               str (for inference output filename)
    """

    def __init__(self, data_root, split_file=None, ids=None,
                 img_size=512, train=True, num_classes=12):
        self.data_root = data_root
        self.img_size = img_size
        self.train = train
        self.num_classes = num_classes

        self.rgb_dir = os.path.join(data_root, "visible")
        self.ir_dir = os.path.join(data_root, "infrared")
        self.depth_dir = os.path.join(data_root, "depth")
        self.label_dir = os.path.join(data_root, "labels")

        if ids is not None:
            self.ids = list(ids)
        elif split_file is not None:
            with open(split_file, "r") as fh:
                self.ids = [ln.strip() for ln in fh if ln.strip()]
        else:
            # Scan labels/ for stems, then keep only those where all 3
            # modality files actually exist. Stray depth-only slices
            # (e.g. "000002_080_00000048") have no labels and are dropped
            # automatically; samples with a label but a missing modality
            # are also dropped here to avoid FileNotFoundError in __getitem__.
            import logging
            logger = logging.getLogger(__name__)
            candidates = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(self.label_dir)
                if f.endswith(".txt")
            )
            kept, dropped = [], []
            for stem in candidates:
                ok = True
                for d in (self.rgb_dir, self.ir_dir, self.depth_dir):
                    try:
                        find_image(stem, d)
                    except FileNotFoundError:
                        ok = False
                        break
                (kept if ok else dropped).append(stem)
            if dropped:
                logger.warning(
                    "Dropped %d/%d samples missing one or more "
                    "modalities (first few: %s)",
                    len(dropped), len(candidates), dropped[:5])
            self.ids = kept

        self.transform = self._build_transform(train)

    def _build_transform(self, train):
        """Geometric augmentation shared by all 3 modalities via additional_targets.

        Photometric normalization is applied per-modality afterwards (RGB ImageNet,
        IR /255, Depth /20000 + mask). Mosaic/MixUp are intentionally disabled:
        they break the cross-modal spatial alignment the fusion relies on.
        """
        if train:
            transforms = [
                _make_transform(
                    A.RandomResizedCrop, self.img_size,
                    scale=(0.5, 1.0), ratio=(0.8, 1.2),
                    interpolation=cv2.INTER_LINEAR, p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        else:
            transforms = [
                _make_transform(A.Resize, self.img_size,
                                interpolation=cv2.INTER_LINEAR),
            ]
        return A.ReplayCompose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.1,
            ),
            additional_targets={"ir": "image", "depth": "image"},
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]

        rgb = np.array(Image.open(find_image(stem, self.rgb_dir)).convert("RGB"))
        ir3 = np.array(Image.open(find_image(stem, self.ir_dir)).convert("RGB"))
        ir = ir3[..., :1]  # 3-channel thermal stack → single channel (spec: visually identical)
        # Depth in this dataset ships in TWO formats:
        #   - 16-bit PNG (mode I;16), values in mm with max≈19999 → /20000
        #   - 8-bit JPG (mode RGB or L), values pre-normalized to [0,255] → /255
        # Detect by dtype/mode and normalize accordingly so both end up in
        # roughly [0,1] with the same physical meaning.
        depth_img = Image.open(find_image(stem, self.depth_dir))
        depth_mode = depth_img.mode
        if depth_mode == "I;16" or depth_mode == "I":
            # 16-bit mm. Keep original dtype to read true uint16 values.
            depth = np.array(depth_img).astype(np.float32)
            depth = depth[..., None] if depth.ndim == 2 else depth[..., :1]
            depth_is_mm = True
        else:
            # 8-bit pre-normalized (JPG or 8-bit PNG). Force to single channel.
            if depth_mode not in ("L",):
                depth_img = depth_img.convert("L")
            depth = np.array(depth_img).astype(np.float32)
            depth = depth[..., None] if depth.ndim == 2 else depth[..., :1]
            depth_is_mm = False

        boxes, class_labels = load_label(
            os.path.join(self.label_dir, stem + ".txt"), self.num_classes)

        H, W = rgb.shape[:2]
        # Boxes are normalized xyxy; convert to pixel coords for albumentations.
        if len(boxes) > 0:
            boxes_px = boxes.copy()
            boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]] * W, 0, W)
            boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]] * H, 0, H)
            box_list = boxes_px.tolist()
            label_list = class_labels.tolist()
        else:
            box_list, label_list = [], []

        data = self.transform(
            image=rgb, ir=ir, depth=depth,
            bboxes=box_list, class_labels=label_list,
        )
        rgb_t = data["image"]
        ir_t = data["ir"]
        depth_t = data["depth"]
        H2, W2 = rgb_t.shape[:2]
        # Albumentations may broadcast a single-channel "image" to 3 channels
        # depending on version/interpolation; force IR and depth back to a
        # single channel so downstream tensors are always (1, H, W).
        if ir_t.ndim == 3 and ir_t.shape[-1] != 1:
            ir_t = ir_t[..., :1]
        if depth_t.ndim == 3 and depth_t.shape[-1] != 1:
            depth_t = depth_t[..., :1]
        elif depth_t.ndim == 2:
            depth_t = depth_t[..., None]

        boxes_aug = np.array(data["bboxes"], dtype=np.float32).reshape(-1, 4)
        labels_aug = np.array(data["class_labels"], dtype=np.int64)
        if len(boxes_aug) > 0:
            boxes_norm = boxes_aug.copy()
            # albumentations returns pascal_voc (x1,y1,x2,y2) pixel coords;
            # normalize to [0,1] keeping xyxy for the clip step.
            boxes_norm[:, [0, 2]] = np.clip(boxes_norm[:, [0, 2]], 0, W2) / W2
            boxes_norm[:, [1, 3]] = np.clip(boxes_norm[:, [1, 3]], 0, H2) / H2
            # DETR's matcher + loss_boxes call center_to_corners_format on both
            # predictions (cxcywh) and targets, so labels MUST be cxcywh too.
            x1, y1, x2, y2 = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
            boxes_norm = np.stack(
                [(x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)], axis=1
            )
        else:
            boxes_norm = np.zeros((0, 4), dtype=np.float32)

        pixel_rgb = self._normalize_rgb(rgb_t)
        pixel_ir = self._normalize_ir(ir_t)
        pixel_depth, depth_mask = self._normalize_depth(depth_t, depth_is_mm)

        target = {
            "class_labels": torch.from_numpy(labels_aug).long(),
            "boxes": torch.from_numpy(boxes_norm).float(),
        }
        return {
            "pixel_values_rgb": pixel_rgb,
            "pixel_values_ir": pixel_ir,
            "pixel_values_depth": pixel_depth,
            "depth_mask": depth_mask,
            "pixel_mask": torch.ones(H2, W2, dtype=torch.bool),
            "labels": target,
            "image_id": torch.tensor(idx, dtype=torch.long),
            "stem": stem,
        }

    @staticmethod
    def _normalize_rgb(rgb_hwc):
        x = rgb_hwc.astype(np.float32) / 255.0
        x = (x - _RGB_MEAN) / _RGB_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

    @staticmethod
    def _normalize_ir(ir_hwc):
        # Single-channel thermal: just scale to [0,1]; cross-modal CSSA handles calibration.
        x = ir_hwc.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

    @staticmethod
    def _normalize_depth(depth_hwc, is_mm=True):
        """Normalize depth to [0,1] and produce a validity mask.

        Two input formats coexist in this dataset:
          - 16-bit mm PNG (is_mm=True):  range [0, 19999] mm, invalid if <10 or >20000.
          - 8-bit pre-normalized JPG (is_mm=False): range [0, 255], no mm
            semantics; treat 0 as invalid but keep the rest.
        Both produce a (1, H, W) float tensor in ~[0,1] and a (H, W) bool mask.
        """
        x = depth_hwc.astype(np.float32)
        if is_mm:
            valid = (x >= 10.0) & (x <= 20000.0)
            x = np.clip(x, 0.0, 20000.0) / 20000.0
        else:
            valid = x > 0.0
            x = x / 255.0
        x[~valid] = 0.0
        tensor = torch.from_numpy(
            np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(valid[..., 0])).bool()
        return tensor, mask


def collate_fn(batch):
    """Stack tensors; keep labels as a list (DETR supports variable targets)."""
    return {
        "pixel_values_rgb": torch.stack([b["pixel_values_rgb"] for b in batch]),
        "pixel_values_ir": torch.stack([b["pixel_values_ir"] for b in batch]),
        "pixel_values_depth": torch.stack([b["pixel_values_depth"] for b in batch]),
        "depth_mask": torch.stack([b["depth_mask"] for b in batch]),
        "pixel_mask": torch.stack([b["pixel_mask"] for b in batch]),
        "image_ids": torch.stack([b["image_id"] for b in batch]),
        "labels": [b["labels"] for b in batch],
        "stems": [b["stem"] for b in batch],
    }

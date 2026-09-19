"""RGB + Infrared + Depth multimodal detection dataset and augmentations.

Directory layout under data_root (per spec):
    rgb/<stem>.{jpg|png}       3-channel 8-bit visible
    ir/<stem>.{jpg|png}        3-channel 8-bit thermal (stacked single channel)
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

# Albumentations 2.x changed geometric transforms: size must be a single
# `size=(h, w)` tuple instead of positional (height, width) ints. Detect once
# and build kwargs accordingly so both 1.x and 2.x work.
try:
    from packaging.version import parse as _parse_ver
    _ALBU_V2 = _parse_ver(A.__version__) >= _parse_ver("2.0")
except Exception:  # pragma: no cover - packaging always present via pip
    _ALBU_V2 = False


def _size_kwargs(size):
    """Return size kwargs compatible with both albumentations 1.x and 2.x."""
    if _ALBU_V2:
        return {"size": (size, size)}
    return {"height": size, "width": size}

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

        self.rgb_dir = os.path.join(data_root, "rgb")
        self.ir_dir = os.path.join(data_root, "ir")
        self.depth_dir = os.path.join(data_root, "depth")
        self.label_dir = os.path.join(data_root, "labels")

        if ids is not None:
            self.ids = list(ids)
        elif split_file is not None:
            with open(split_file, "r") as fh:
                self.ids = [ln.strip() for ln in fh if ln.strip()]
        else:
            self.ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(self.label_dir)
                if f.endswith(".txt")
            )

        self.transform = self._build_transform(train)

    def _build_transform(self, train):
        """Geometric augmentation shared by all 3 modalities via additional_targets.

        Photometric normalization is applied per-modality afterwards (RGB ImageNet,
        IR /255, Depth /20000 + mask). Mosaic/MixUp are intentionally disabled:
        they break the cross-modal spatial alignment the fusion relies on.
        """
        if train:
            transforms = [
                A.RandomResizedCrop(
                    **_size_kwargs(self.img_size),
                    scale=(0.5, 1.0), ratio=(0.8, 1.2),
                    interpolation=cv2.INTER_LINEAR, p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        else:
            transforms = [
                A.Resize(**_size_kwargs(self.img_size),
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
        depth = np.array(Image.open(find_image(stem, self.depth_dir))).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[..., None]

        boxes, class_labels = load_label(
            os.path.join(self.label_dir, stem + ".txt"), self.num_classes)

        H, W = rgb.shape[:2]
        # Boxes are normalized xyxy; convert to pixel coords for albumentations.
        if len(boxes) > 0:
            boxes_px = boxes.copy()
            boxes_px[:, [0, 2]] *= W
            boxes_px[:, [1, 3]] *= H
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
        pixel_depth, depth_mask = self._normalize_depth(depth_t)

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
    def _normalize_depth(depth_hwc):
        # uint16 mm → [0,1]. Spec: 0 or too small = invalid; range [0, 19999]mm.
        x = depth_hwc.astype(np.float32)
        valid = (x >= 10.0) & (x <= 20000.0)
        x = np.clip(x, 0.0, 20000.0) / 20000.0
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

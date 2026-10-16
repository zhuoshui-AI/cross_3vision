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
                 img_size=512, train=True, num_classes=12,
                 scan_source="labels"):
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
            # Scan for stems. scan_source="labels" (default, training):
            # enumerate labels/*.txt — every training sample must be labeled.
            # scan_source="images" (inference on unlabeled test sets):
            # enumerate visible/ image files; per-sample label txts are
            # optional (load_label returns empty for missing files).
            import logging
            logger = logging.getLogger(__name__)
            if scan_source == "images":
                candidates = sorted(
                    os.path.splitext(f)[0]
                    for f in os.listdir(self.rgb_dir)
                    if f.lower().endswith(IMG_EXTS)
                )
            else:
                if not os.path.isdir(self.label_dir):
                    raise FileNotFoundError(
                        f"labels dir not found: {self.label_dir} — training "
                        "needs labels; for unlabeled test sets construct "
                        "MultiModalDataset(scan_source='images')")
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

        正则化（治过拟合，10-01 实验证据）：crop 下限 0.5→0.4 增强尺度扰动；
        另在 __getitem__ 里对 RGB 单独做光度增强、对 IR 做热图对比度/噪声
        增强、对深度做传感器噪声/脱落模拟 —— albumentations 的
        additional_targets 会把光度变换同等地作用到所有模态上，而热图
        和深度是物理量，各模态的光度增强必须手工单独施加。

        深度 validity mask 以 additional_target "depth_valid"（mask 类型，
        最近邻插值）与深度图共享同一套几何变换，保证像素级对齐。
        """
        if train:
            transforms = [
                _make_transform(
                    A.RandomResizedCrop, self.img_size,
                    scale=(0.4, 1.0), ratio=(0.8, 1.2),
                    interpolation=cv2.INTER_LINEAR, p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
            ]
            # ---- 训练期各模态光度增强器（仅 train 生效）----
            # CLAHE 对象复用于 RGB-L 通道与 IR 单通道（clip=2.0, tile=8x8）
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self._shadow = A.RandomShadow(p=0.25)
            # 触发概率集中管理（消融时改这里）
            self._p_rgb_clahe = 0.30
            self._p_ir_clahe = 0.30     # 热图局部对比度：弱热信号目标更醒目
            self._p_ir_noise = 0.30     # 高斯噪声：模拟传感器热噪声
            self._p_ir_denoise = 0.15   # 中值滤波：模拟"干净/模糊"传感器两态
            self._p_depth_noise = 0.30  # 有效像素高斯噪声（20~80mm 量级）
            self._p_depth_dropout = 0.20  # 随机深度脱落（模拟传感器空洞）
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
            additional_targets={"ir": "image", "depth": "image",
                                "depth_valid": "mask"},
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

        # ---- 深度空洞邻域填充（确定性预处理：train/val/推理三阶段一致）----
        # 无效像素（16bit: <10mm / >20000mm 哨兵；8bit: 0）在 resize 线性
        # 插值时会把 0 值拖进有效区边缘形成假梯度；先做 3x3 邻域均值外扩
        # 填充。填充像素在 validity mask 中仍标记无效 —— 合成值只给
        # DepthEncoder CNN 当局部上下文，不作为真值进入深度交叉注意力。
        if depth_is_mm:
            depth_invalid = (depth[..., 0] < 10.0) | (depth[..., 0] > 20000.0)
        else:
            depth_invalid = depth[..., 0] <= 0.0
        invalid_frac = float(depth_invalid.mean())
        if 0.0 < invalid_frac <= 0.5:  # 全无效样本跳过（mask 已兜底）
            filled = self._fill_depth_holes(depth[..., 0], depth_invalid)
            depth = filled[..., None]
        # validity mask（255=有效）作为 mask 类型的 additional_target，
        # 与深度图共享几何变换（最近邻插值，不做平滑）
        depth_valid_u8 = (~depth_invalid).astype(np.uint8) * 255

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
            image=rgb, ir=ir, depth=depth, depth_valid=depth_valid_u8,
            bboxes=box_list, class_labels=label_list,
        )
        rgb_t = data["image"]
        ir_t = data["ir"]
        depth_t = data["depth"]
        # 变换后的 validity mask（mask 类型走最近邻插值，值域仍为 0/255）
        dv = np.asarray(data["depth_valid"])
        depth_valid = dv.astype(bool) if dv.dtype == bool else dv > 127

        # ---- 训练期光度增强（各模态独立；val/推理不执行）----
        if self.train:
            # RGB：亮度/对比度抖动 + CLAHE + 随机阴影
            rgb_t = self._jitter_rgb(rgb_t)
            rgb_t = self._augment_rgb_train(rgb_t)
            # IR：CLAHE（热图局部对比度，弱热信号目标更醒目）
            #      + 高斯噪声（模拟传感器热噪声）
            #      + 中值滤波（模拟干净/模糊传感器两态）
            ir_t = self._augment_ir_train(ir_t)
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
        pixel_depth, depth_mask = self._normalize_depth(
            depth_t, depth_is_mm, depth_valid)
        # 深度传感器噪声/脱落模拟（仅训练期）
        if self.train:
            pixel_depth, depth_mask = self._augment_depth_train(
                pixel_depth, depth_mask)

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
    def _jitter_rgb(rgb_hwc):
        """轻度亮度/对比度抖动，仅用于训练期 RGB（保持 uint8 输出）。

        幅度刻意温和（对比度 ±10%、亮度 ±15/255）：目标是打掉像素级
        记忆化（过拟合源头之一），不是模拟光照剧变。
        """
        contrast = 1.0 + float(np.random.uniform(-0.1, 0.1))
        brightness = float(np.random.uniform(-15.0, 15.0))
        x = (rgb_hwc.astype(np.float32) - 128.0) * contrast + 128.0 + brightness
        return np.clip(x, 0.0, 255.0).astype(np.uint8)

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
    def _fill_depth_holes(d2d, invalid, max_iters=32):
        """3x3 邻域迭代均值填充深度空洞（确定性，cv2 实现，无 scipy 依赖）。

        每轮把"已填充集合"外扩 1px：新填像素取其 8 邻域中已有效像素的
        均值（等价于低通扩散插值），空洞半径 r 需约 r 轮。返回值仅填充
        像素变化；invalid 掩码由调用方保留（填充像素仍标记无效）。
        超过 max_iters 仍未填到的大洞保持原值（mask=0 已兜底）。
        """
        if not invalid.any() or invalid.all():
            return d2d.astype(np.float32, copy=True)
        out = d2d.astype(np.float32).copy()
        filled = (~invalid).astype(np.float32)
        todo = invalid.copy()
        ones = np.ones((3, 3), np.float32)
        for _ in range(max_iters):
            if not todo.any():
                break
            cnt = cv2.filter2D(filled, -1, ones,
                               borderType=cv2.BORDER_CONSTANT)
            region = todo & (cnt > 0)
            if not region.any():
                break
            ssum = cv2.filter2D(out * filled, -1, ones,
                                borderType=cv2.BORDER_CONSTANT)
            out[region] = ssum[region] / cnt[region]
            filled[region] = 1.0
            todo = ~filled.astype(bool)
        return out

    def _augment_rgb_train(self, rgb_hwc):
        """训练期 RGB 光度增强：CLAHE（LAB-L 通道）+ 随机阴影。

        CLAHE 限制局部对比度增益（clip=2.0），不会像全局直方图均衡那样
        放大噪声；随机阴影模拟户外光照分布，两者均为温和幅度。
        """
        if np.random.rand() < self._p_rgb_clahe:
            lab = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            rgb_hwc = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # RandomShadow 构造时已带 p=0.25，内部自行掷骰
        rgb_hwc = self._shadow(image=rgb_hwc)["image"]
        return rgb_hwc

    def _augment_ir_train(self, ir_hwc):
        """训练期 IR 热图增强（单通道 uint8）。

        CLAHE：热图常有大片低对比区域，弱热信号目标（远处的人/uav）
        局部对比度拉伸后更醒目；高斯噪声模拟传感器热噪声；中值滤波
        模拟"干净/模糊"两种传感器状态，教会模型对模糊鲁棒。
        """
        c = ir_hwc[..., 0] if ir_hwc.ndim == 3 else ir_hwc
        if np.random.rand() < self._p_ir_clahe:
            c = self._clahe.apply(c)
        if np.random.rand() < self._p_ir_denoise:
            c = cv2.medianBlur(c, 3)
        if np.random.rand() < self._p_ir_noise:
            sigma = float(np.random.uniform(5.0, 15.0))
            c = np.clip(
                c.astype(np.float32)
                + np.random.normal(0.0, sigma, c.shape), 0, 255
            ).astype(np.uint8)
        return c[..., None]

    @staticmethod
    def _augment_depth_train(depth_chw, valid_mask):
        """训练期深度传感器噪声/脱落模拟（归一化后施加）。

        1) 高斯噪声：仅在有效像素上，σ 取量程的 0.1%~0.4%（≈20~80mm），
           模拟深度传感器的测量抖动；
        2) 随机脱落：2~5 个小矩形区域整体置无效，模拟传感器空洞/反光
           失效——推理时的深度图经常出现这类缺失，训练期见过才能鲁棒。
        """
        if torch.rand(()) < 0.30:
            sigma = float(torch.empty(1).uniform_(0.001, 0.004))
            noisy = (depth_chw + torch.randn_like(depth_chw) * sigma
                     ).clamp(0.0, 1.0)
            depth_chw = torch.where(valid_mask, noisy, depth_chw)
        if torch.rand(()) < 0.20:
            _, H, W = depth_chw.shape
            n_holes = int(torch.randint(2, 6, ()))
            for _ in range(n_holes):
                rw = int(torch.randint(max(2, W // 50), max(3, W // 12) + 1, ()))
                rh = int(torch.randint(max(2, H // 50), max(3, H // 12) + 1, ()))
                x0 = int(torch.randint(0, max(1, W - rw), ()))
                y0 = int(torch.randint(0, max(1, H - rh), ()))
                valid_mask[y0:y0 + rh, x0:x0 + rw] = False
                depth_chw[:, y0:y0 + rh, x0:x0 + rw] = 0.0
        return depth_chw, valid_mask

    @staticmethod
    def _normalize_depth(depth_hwc, is_mm=True, valid=None):
        """Normalize depth to [0,1]; invalid pixels zeroed.

        Two input formats coexist in this dataset:
          - 16-bit mm PNG (is_mm=True):  range [0, 19999] mm, invalid if <10 or >20000.
          - 8-bit pre-normalized JPG (is_mm=False): range [0, 255], no mm
            semantics; treat 0 as invalid but keep the rest.
        valid: 变换后的 (H, W) bool 掩码（与深度图共享几何变换）；
               为 None 时按值域推断（兼容旧调用方式）。
        Both produce a (1, H, W) float tensor in ~[0,1] and a (H, W) bool mask.
        """
        x = depth_hwc.astype(np.float32)
        if is_mm:
            x = np.clip(x, 0.0, 20000.0) / 20000.0
        else:
            x = x / 255.0
        if valid is None:
            if is_mm:
                valid = (depth_hwc[..., 0] >= 10.0) & (depth_hwc[..., 0] <= 20000.0)
            else:
                valid = depth_hwc[..., 0] > 0.0
        x[~valid] = 0.0
        tensor = torch.from_numpy(
            np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(valid)).bool()
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

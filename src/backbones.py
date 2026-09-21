"""Per-modality feature extractors for RGB, Infrared, and Depth.

- SwinRGBBackbone: timm Swin (ImageNet-22k pretrained) → stride-32 feature map.
- IRBackbone: thermal-pretrained fallback (ImageNet ResNet50 adapted to 1-channel
  input by summing weights across the original 3 input channels) → stride-32 map,
  projected to match Swin's channel count.
- DepthEncoder: lightweight 5-stride-2 CNN, depth + validity mask as 2-channel
  input; no pretraining (depth is geometric signal, trained from 0).
"""
import torch
import torch.nn as nn
import timm
import torchvision


def _conv_bn_act(in_c, out_c, k=3, s=2, p=1):
    """Conv-BN-SiLU block with stride s (GroupNorm to avoid batch-size sensitivity)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


def _group_norm(channels, groups=32):
    """GroupNorm with a group count that always divides `channels`.

    GroupNorm replaces BatchNorm in the IR backbone so that behaviour is
    IDENTICAL in train/eval (BN switches batch-stats ↔ running-stats, which is
    catastrophic here: conv1 was summed 3ch→1ch but bn1 kept 3ch ImageNet
    running stats, and per-GPU batch size is small).
    """
    g = groups
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


def convert_msft_swin_to_timm(state):
    """Remap official Microsoft Swin checkpoint keys to modern timm layout.

    Tensors are identical; only the stage index of PatchMerging differs:
      official: layers.{0,1,2}.downsample.*   (downsample at end of stage i)
      timm    : layers.{1,2,3}.downsample.*   (downsample at start of stage i+1)
    Block / patch_embed / norm keys stay the same.
    """
    remapped = {}
    for k, v in state.items():
        nk = k
        if k.startswith("layers.0.downsample."):
            nk = "layers.1.downsample." + k[len("layers.0.downsample."):]
        elif k.startswith("layers.1.downsample."):
            nk = "layers.2.downsample." + k[len("layers.1.downsample."):]
        elif k.startswith("layers.2.downsample."):
            nk = "layers.3.downsample." + k[len("layers.2.downsample."):]
        remapped[nk] = v
    return remapped


class SwinRGBBackbone(nn.Module):
    """Swin Transformer (ImageNet-22k) producing stride-32 features.

    Output: (B, swin_dim, H/32, W/32). For Swin-S swin_dim=768, Swin-B=1024.
    """

    def __init__(self, variant="swin_small_patch4_window7_224_22k",
                 pretrained=True, pretrained_path=None, freeze_stages=2,
                 img_size=512, pretrained_tag="ms_in22k", drop_path_rate=0.0):
        super().__init__()
        import os
        path_ok = bool(pretrained_path) and os.path.isfile(pretrained_path)
        if pretrained_path and not path_ok:
            import logging
            logging.getLogger(__name__).warning(
                "Swin pretrained_path '%s' not found; falling back to "
                "unpretrained weights. Run scripts/download_pretrained.sh.",
                pretrained_path)
        # Fetch from timm hub only when a local file isn't supplied.
        kwargs = {"pretrained": pretrained and not path_ok}
        if pretrained_tag and kwargs.get("pretrained"):
            # e.g. "ms_in22k" → ImageNet-22k weights (user's stated requirement).
            kwargs["pretrained_cfg"] = pretrained_tag
        # drop_path_rate=0 keeps train/eval forward identical (timm's default
        # for swin_small is 0.3, which adds stochastic residual drops).
        kwargs["drop_path_rate"] = float(drop_path_rate)
        # Pass img_size so Swin precomputes its window-attention mask for the
        # real training/inference resolution (the mask is resolution-locked).
        try:
            self.swin = timm.create_model(variant, img_size=img_size, **kwargs)
        except TypeError:
            # Older timm without img_size kwarg support.
            kwargs.pop("pretrained_cfg", None)
            kwargs.pop("drop_path_rate", None)
            self.swin = timm.create_model(variant, **kwargs)
        if path_ok:
            import logging
            state = torch.load(pretrained_path, map_location="cpu")
            if "model" in state:
                state = state["model"]
            state = convert_msft_swin_to_timm(state)
            result = self.swin.load_state_dict(state, strict=False)
            # Every backbone tensor must match; the only tolerated leftovers
            # are the 22k classifier head and the precomputed attn-mask
            # buffers. Shape mismatches must be surfaced, not silently skipped.
            model_keys = set(self.swin.state_dict().keys())
            ck_keys = set(state.keys())

            def _is_head_or_buffer(k):
                return (k.startswith("head.")
                        or "attn_mask" in k
                        or k.startswith("patch_embed.norm"))

            shape_bad = [k for k in (ck_keys & model_keys)
                         if state[k].shape != self.swin.state_dict()[k].shape]
            missing_backbone = [
                k for k in (model_keys - ck_keys)
                if not _is_head_or_buffer(k)]
            logger = logging.getLogger(__name__)
            if shape_bad:
                logger.error("Swin checkpoint shape mismatches: %s", shape_bad)
            if missing_backbone:
                logger.warning(
                    "Swin checkpoint missing backbone keys (random-init): %s",
                    missing_backbone[:10])
            logger.info(
                "Loaded Swin 22k checkpoint: %d tensors matched "
                "(unexpected leftovers: %d, e.g. %s)",
                len(ck_keys & model_keys) - len(shape_bad),
                len(result.unexpected_keys),
                result.unexpected_keys[:5])
        # Swin uses relative position bias (not absolute), so it accepts
        # arbitrary input sizes as long as they divide the patch size. The
        # stock PatchEmbed asserts H/W == img_size; disable that so we can
        # feed other sizes (e.g. during inference) without crashing.
        if hasattr(self.swin, "patch_embed"):
            self.swin.patch_embed.strict_img_size = False
        self.swin_dim = self.swin.num_features  # 768 (Swin-S)
        self.freeze_stages = freeze_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze patch embed + first N stages to save memory / avoid destabilising."""
        if self.freeze_stages >= 1:
            for p in self.swin.patch_embed.parameters():
                p.requires_grad = False
            if hasattr(self.swin, "pos_drop"):
                for p in self.swin.pos_drop.parameters():
                    p.requires_grad = False
            if hasattr(self.swin, "norm_pre"):
                for p in self.swin.norm_pre.parameters():
                    p.requires_grad = False
        for i, layer in enumerate(self.swin.layers):
            if i < self.freeze_stages:
                for p in layer.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """Frozen stages stay in eval() even during training.

        model.train() flips every submodule to train mode; that would silently
        re-enable dropout/DropPath inside FROZEN stages, feeding stochastic
        features into the trainable stages while eval runs them clean — a
        train/eval distribution mismatch. Pin frozen submodules to eval.
        """
        super().train(mode)
        if mode and self.freeze_stages >= 1:
            self.swin.patch_embed.eval()
            if hasattr(self.swin, "pos_drop"):
                self.swin.pos_drop.eval()
            if hasattr(self.swin, "norm_pre"):
                self.swin.norm_pre.eval()
            for i, layer in enumerate(self.swin.layers):
                if i < self.freeze_stages:
                    layer.eval()
        return self

    def forward(self, x):
        """x: (B, 3, H, W) ImageNet-normalized. Returns (B, C, H/32, W/32)."""
        B, _, H, W = x.shape
        # Manual iteration to grab the final-stage spatial feature map.
        feats = self.swin.patch_embed(x)
        if hasattr(self.swin, "pos_drop"):
            feats = self.swin.pos_drop(feats)
        if hasattr(self.swin, "norm_pre"):
            feats = self.swin.norm_pre(feats)
        for layer in self.swin.layers:
            feats = layer(feats)
        # feats: (B, L, C) with L = (H/32) * (W/32).
        Hp, Wp = H // 32, W // 32
        feats = feats.transpose(1, 2).reshape(B, self.swin_dim, Hp, Wp).contiguous()
        return feats


class IRBackbone(nn.Module):
    """Infrared branch.

    Fallback mode: ImageNet ResNet50 with conv1 adapted to 1-channel input by
    summing pretrained weights across the original 3 input channels — equivalent
    to feeding the IR grayscale through all 3 RGB weights and summing. A
    1x1 conv projects layer4 (2048-dim, stride 32) to match Swin's channel dim.

    Switching to AnyThermal (DINOv2-distilled ViT-B/14 thermal backbone) is a
    drop-in via `variant="anythermal"` once weights are downloaded; that path
    reshapes patch tokens and applies a stride adapter to land on H/32.
    """

    def __init__(self, out_dim, variant="resnet50_imagenet_1ch",
                 pretrained_path=None, anythermal_path=None):
        super().__init__()
        self.variant = variant
        self.out_dim = out_dim

        if variant == "anythermal":
            self.backbone = _AnyThermalAdapter(anythermal_path, out_dim=out_dim)
            self.in_channels = self.backbone.out_channels
        elif variant == "resnet50_imagenet_1ch":
            # Build with GroupNorm everywhere (no train/eval mode switch), then
            # copy the pretrained BN affine params (γ/β share the same key
            # names); running_mean/var and num_batches_tracked are discarded.
            rn = torchvision.models.resnet50(
                weights=None, norm_layer=_group_norm)
            if pretrained_path:
                state = torch.load(pretrained_path, map_location="cpu")
                rn.load_state_dict(state, strict=False)
            else:
                # Online V2 weights ship as BN tensors; same γ/β copy applies.
                rn_bn = torchvision.models.resnet50(weights="IMAGENET1K_V2")
                rn.load_state_dict(rn_bn.state_dict(), strict=False)
                del rn_bn
            # Adapt conv1 from 3ch→1ch by summing weights across input channels.
            with torch.no_grad():
                new_w = rn.conv1.weight.sum(dim=1, keepdim=True)  # (64, 1, 7, 7)
            rn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                rn.conv1.weight.copy_(new_w)
            # Keep only the parts we need (through layer4).
            self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
            self.layer1 = rn.layer1
            self.layer2 = rn.layer2
            self.layer3 = rn.layer3
            self.layer4 = rn.layer4
            self.in_channels = 2048
        else:
            raise ValueError(f"Unknown IR variant: {variant}")

        # Adapter: project to Swin's channel dim + GroupNorm.
        self.adapter = nn.Sequential(
            nn.Conv2d(self.in_channels, out_dim, 1, bias=False),
            nn.GroupNorm(8, out_dim),
        )

    def forward(self, x):
        """x: (B, 1, H, W) IR (single channel). Returns (B, out_dim, H/32, W/32)."""
        if self.variant == "anythermal":
            # Placeholder until AnyThermal weights are wired in. Reshaping +
            # stride-2 interpolation to H/32 will happen here.
            f = self.backbone(x)
            target_hw = (x.shape[-2] // 32, x.shape[-1] // 32)
            if f.shape[-2:] != target_hw:
                f = nn.functional.interpolate(
                    f, size=target_hw, mode="bilinear", align_corners=False)
        else:
            f = self.stem(x)
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f)  # (B, 2048, H/32, W/32)
        return self.adapter(f)


class _AnyThermalAdapter(nn.Module):
    """Thin wrapper around AnyThermal ViT-B/14 (placeholder for swap-in later).

    The actual AnyThermal checkpoint is loaded lazily; if absent this module
    raises a clear error pointing to scripts/download_pretrained.sh. The
    structure here mirrors how the swap-in will reshape patch tokens and apply
    a stride-2 adapter to land on H/32.
    """

    def __init__(self, weights_path, out_dim=768):
        super().__init__()
        self.weights_path = weights_path
        self.out_channels = out_dim
        self._loaded = False
        self._backbone = None
        # Stride adapter: ViT-B/14 patch stride 14 → upsample/downsample to /32.
        self.stride_adapter = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(inplace=True),
        )

    def _load(self):
        if self._loaded:
            return
        if not self.weights_path:
            raise RuntimeError(
                "AnyThermal requested but no weights path set. Run "
                "scripts/download_pretrained.sh WITH_ANYTHERMAL=1 first, or "
                "fall back to variant='resnet50_imagenet_1ch' in the config."
            )
        # Placeholder: real AnyThermal loader belongs here once the swap-in is
        # done. Until then, raise to surface the missing dependency cleanly.
        raise RuntimeError(
            "AnyThermal integration pending — use variant='resnet50_imagenet_1ch' "
            "fallback first to validate the pipeline end-to-end."
        )

    def forward(self, x):
        self._load()
        # Real path will be: tokenize → ViT blocks → reshape (B, C, H/14, W/14) →
        # interpolate to (H/32, W/32) → stride_adapter.
        raise RuntimeError("AnyThermal forward not yet wired.")

    def __call__(self, x):
        return self.forward(x)


class DepthEncoder(nn.Module):
    """Lightweight CNN encoding 16-bit depth + validity mask → stride-32 features.

    Input: depth (B, 1, H, W) float in [0,1] (20000mm mapped) and mask (B, H, W)
    bool. The mask is concatenated as a second channel so the encoder can learn
    to ignore invalid regions directly; DepthLateFusion also receives the
    downsampled mask as key_padding_mask for cross-attention.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            _conv_bn_act(2, 16, k=3, s=2, p=1),    # /2
            _conv_bn_act(16, 32, k=3, s=2, p=1),   # /4
            _conv_bn_act(32, 64, k=3, s=2, p=1),   # /8
            _conv_bn_act(64, 128, k=3, s=2, p=1),  # /16
            _conv_bn_act(128, out_dim, k=3, s=2, p=1),  # /32
        )
        self.out_dim = out_dim

    def forward(self, depth, depth_mask):
        # depth: (B,1,H,W), depth_mask: (B,H,W) bool
        x = torch.cat([depth, depth_mask.float().unsqueeze(1)], dim=1)
        return self.conv(x)

    @staticmethod
    def downsample_mask(depth_mask, target_hw):
        """Downsample the validity mask to feature-map size for cross-attention padding mask."""
        # (B, H, W) → (B, h, w) using max-pool (any valid pixel in window → valid).
        m = depth_mask.float().unsqueeze(1)  # (B,1,H,W)
        m = nn.functional.adaptive_max_pool2d(m, target_hw)
        return m.squeeze(1).bool()  # (B, h, w)

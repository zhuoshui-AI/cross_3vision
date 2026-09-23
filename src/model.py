"""Multi-modal Swin + DETR detection model.

Architecture (per approved plan):
    1) Front-half: RGB → Swin (ImageNet-22k); IR → ResNet50-1ch fallback → adapter
       to Swin dim; CSSA channel-switching spatial-attention fuses RGB↔IR, projects
       to DETR d_model.
    2) Back-half: depth (16-bit mm + validity mask) → lightweight CNN; single
       cross-attention layer injects depth into the fused RGB+IR stream (Q=fused,
       K=V=depth, depth_mask → key_padding_mask).
    3) DETR encoder/decoder + class/box heads reused verbatim from HuggingFace
       `DetrForObjectDetection` (matcher, loss, auxiliary losses, post-processing).

We subclass `DetrForObjectDetection` and only replace the conv-encoder part of
`DetrModel.backbone` with our `MultiModalBackbone`, then override `forward` to
feed a dict of per-modality tensors instead of a single `pixel_values` tensor
(the stock `DetrModel.forward` starts with `pixel_values.shape`, which breaks on
a dict). Everything downstream (input_projection, sine pos-embed, encoder,
decoder, heads, loss_function) is reused as-is.

`self.loss_function` is a property on `PreTrainedModel` that resolves
`loss_type` against `LOSS_MAPPING` by regex-matching the class name. Our class
name `MultiModalSwinDETR` matches none of the keys, so we explicitly set
`self.loss_type = "ForObjectDetection"` in `__init__` to route to
`ForObjectDetectionLoss` (Hungarian matcher + CE/L1/GIoU).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DetrConfig, DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

from .backbones import (
    SwinRGBBackbone, IRBackbone, DepthEncoder, SwinModalityBackbone)
from .fusion import CSSAFusion, DepthLateFusion, HCMAFBlock, SimpleFPN


class MultiModalBackbone(nn.Module):
    """Replaces `DetrConvEncoder` inside `DetrModel.backbone`.

    forward(pixel_values, pixel_mask) mirrors DetrConvEncoder.forward's return
    contract: a list of `(feature_map, mask)` tuples (we emit a single tuple).
    DetrConvModel then computes position embeddings from it — we leave that
    logic intact, so only the conv_encoder slot is swapped.

    pixel_values is a dict (NOT a tensor):
        rgb    (B, 3, H, W) ImageNet-normalized
        ir     (B, 1, H, W) [0,1]
        depth  (B, 1, H, W) [0,1] (20000 mm mapped), 0 where invalid
        depth_mask (B, H, W) bool, True where depth is valid
    """

    intermediate_channel_sizes = [256]  # compat attr accessed by DetrModel.__init__

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        d_model = int(mcfg["d_model"])

        # RGB branch: Swin (ImageNet-22k). swin_dim (768 for Swin-S) is the
        # CSSA channel count; IR adapter projects to the same dim. We default
        # pretrained=False at construct time to avoid a network fetch during
        # model build; train.py loads weights via timm hub (ms_in22k tag) or
        # via the explicit pretrained_path set in config.
        self.rgb_backbone = SwinRGBBackbone(
            variant=mcfg["swin_variant"],
            pretrained=bool(mcfg.get("swin_pretrained", False)),
            pretrained_path=mcfg.get("swin_pretrained_path") or None,
            freeze_stages=int(mcfg.get("swin_freeze_stages", 2)),
            img_size=int(cfg["data"]["img_size"]),
            pretrained_tag=mcfg.get("swin_pretrained_tag", "ms_in22k"),
            drop_path_rate=float(mcfg.get("swin_drop_path_rate", 0.0)),
        )
        swin_dim = self.rgb_backbone.swin_dim

        # IR branch: ResNet50-1ch fallback (default) or AnyThermal (placeholder).
        self.ir_backbone = IRBackbone(
            out_dim=swin_dim,
            variant=mcfg.get("ir_backbone", "resnet50_imagenet_1ch"),
            pretrained_path=mcfg.get("ir_pretrained_path") or None,
            anythermal_path=mcfg.get("ir_pretrained_path") or None,
        )

        # Front-half fusion: CSSA switches low-scoring channels between RGB and
        # IR, spatial-gates the mix, projects swin_dim → d_model.
        self.cssa = CSSAFusion(
            channels=swin_dim,
            out_dim=d_model,
            tau=float(mcfg.get("cssa_tau", 0.5)),
            reduction=int(mcfg.get("cssa_reduction", 16)),
        )

        # Depth branch: 2-channel (depth + mask) → stride-32 features at d_model.
        self.depth_encoder = DepthEncoder(out_dim=d_model)

        # Back-half fusion: single cross-attn injecting depth into fused RGB+IR.
        self.depth_fuse = DepthLateFusion(
            d_model=d_model,
            num_heads=int(mcfg.get("depth_fuse_heads", 8)),
            dropout=float(mcfg.get("depth_fuse_dropout", 0.0)),
        )

        self.d_model = d_model

    def forward(self, pixel_values, pixel_mask):
        rgb = pixel_values["rgb"]
        ir = pixel_values["ir"]
        depth = pixel_values["depth"]
        depth_mask = pixel_values["depth_mask"]

        # Stride-32 feature maps from each branch.
        f_rgb = self.rgb_backbone(rgb)            # (B, swin_dim, h, w)
        f_ir = self.ir_backbone(ir)              # (B, swin_dim, h, w)
        f_fused = self.cssa(f_rgb, f_ir)         # (B, d_model, h, w)
        f_depth = self.depth_encoder(depth, depth_mask)  # (B, d_model, h, w)
        depth_mask_ds = DepthEncoder.downsample_mask(
            depth_mask, f_fused.shape[-2:])     # (B, h, w) bool
        f_enh = self.depth_fuse(f_fused, f_depth, depth_mask_ds)  # (B, d_model, h, w)

        # Downsample pixel_mask to feature-map size (mirrors DetrConvEncoder).
        if pixel_mask is None:
            mask_ds = torch.ones(
                (f_enh.shape[0], f_enh.shape[2], f_enh.shape[3]),
                dtype=torch.bool, device=f_enh.device)
        else:
            mask_ds = F.interpolate(
                pixel_mask[None].float(), size=f_enh.shape[-2:]
            ).to(torch.bool)[0]
        return [(f_enh, mask_ds)]


class MultiModalSwinDETR(DetrForObjectDetection):
    """DetrForObjectDetection with a multi-modal backbone (RGB+IR+Depth).

    Stock DetrForObjectDetection forward expects `pixel_values` to be a tensor
    (it does `pixel_values.shape`); we accept per-modality tensors and drive
    the encoder/decoder pipeline manually. Loss + heads are inherited.
    """

    def __init__(self, config: DetrConfig, cfg=None):
        super().__init__(config)
        # Force the loss-function property to resolve to ForObjectDetectionLoss.
        # Without this, our class name matches no LOSS_MAPPING key and the
        # property falls back to ForCausalLMLoss (wrong task).
        self.loss_type = "ForObjectDetection"

        d_model = config.d_model
        # Our backbone emits d_model channels (not resnet50's 2048); rebuild
        # the 1x1 projection so the channel count matches.
        self.model.input_projection = nn.Conv2d(
            d_model, d_model, kernel_size=1)
        # Swap the conv-encoder; DetrConvModel keeps its position-embedding logic.
        if cfg is None:
            raise ValueError("MultiModalSwinDETR needs the run config `cfg`.")
        self.model.backbone.conv_encoder = MultiModalBackbone(cfg)
        # Initialize only the freshly-created input_projection.
        # IMPORTANT: do NOT call self.post_init() here. super().__init__ already
        # initialized the DETR encoder/decoder/heads, and MultiModalBackbone
        # loaded pretrained Swin/ResNet weights internally. A second post_init()
        # would recursively reinitialize every Linear/Conv2d with N(0, 0.02),
        # destroying the pretrained backbone weights and forcing the model to
        # train from scratch (which is why mAP stayed at 0 for many epochs).
        self.model.input_projection.apply(self._init_weights)

        # DETR foreground-prior bias init (Carion et al. ECCV 2020, §A.4):
        # set the final classification-layer bias to -log((1-pi)/pi) so the
        # model initially predicts objects with probability ~pi. Without this,
        # all 100 queries start at ~1/num_classes confidence for every class,
        # the Hungarian matcher sees dense false positives, and training
        # oscillates (ce bounces 0.5<->1.2, giou never converges). pi=0.01
        # matches the original paper; we keep it class-agnostic.
        import math
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        with torch.no_grad():
            self.class_labels_classifier.bias.fill_(bias_value)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        pixel_values_rgb,
        pixel_values_ir,
        pixel_values_depth,
        depth_mask,
        pixel_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Manual pipeline bypassing DetrModel.forward's `pixel_values.shape` line.

        Args mirror what collate_fn / inference produces. pixel_values_* are
        (B, C, H, W) tensors; depth_mask & pixel_mask are (B, H, W) bool.
        """
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        device = pixel_values_rgb.device
        batch_size = pixel_values_rgb.shape[0]
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, pixel_values_rgb.shape[2], pixel_values_rgb.shape[3]),
                dtype=torch.bool, device=device)

        pixel_values = {
            "rgb": pixel_values_rgb,
            "ir": pixel_values_ir,
            "depth": pixel_values_depth,
            "depth_mask": depth_mask,
        }

        # Backbone: returns (features_list, pos_list); each features item is
        # (feature_map, mask). Mirrors DetrConvModel.forward.
        features, object_queries_list = self.model.backbone(
            pixel_values, pixel_mask)
        feature_map, mask = features[-1]
        if mask is None:
            raise ValueError("Backbone did not return a downsampled pixel mask")

        # 1x1 channel projection → flatten to (B, HW, d_model).
        projected = self.model.input_projection(feature_map)
        flattened = projected.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        flattened_mask = mask.flatten(1)

        encoder_outputs = self.model.encoder(
            inputs_embeds=flattened,
            attention_mask=flattened_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        query_pos = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1)
        queries = torch.zeros_like(query_pos)
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_pos,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = (decoder_outputs.intermediate_hidden_states
                                if return_dict else decoder_outputs[4])
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, device, pred_boxes, self.config,
                outputs_class, outputs_coord)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + decoder_outputs + encoder_outputs
            else:
                output = (logits, pred_boxes) + decoder_outputs + encoder_outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # ----------------------------------------------------------- backbone freeze
    def freeze_backbone(self):
        """Freeze the per-modality backbones (Swin, IR, depth) but keep the
        fusion modules and DETR heads trainable."""
        enc = self.model.backbone.conv_encoder
        for sub in (enc.rgb_backbone, enc.ir_backbone, enc.depth_encoder):
            for p in sub.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        enc = self.model.backbone.conv_encoder
        for sub in (enc.rgb_backbone, enc.ir_backbone, enc.depth_encoder):
            for p in sub.parameters():
                p.requires_grad = True


class TriModalSwinFusion(nn.Module):
    """Three per-modality Swin towers + per-scale HCMAF + FPN.

    Pipeline:
      RGB (3ch)   ─┐
      IR  (1ch)   ─┼─ Swin stage i ── HCMAF block (fused stages only) ── next stage
      Depth (2ch) ─┘
    Fused maps at fuse stages (default stages 2/3/4 → stride 8/16/32) go
    through SimpleFPN → P3/P4/P5 at d_model channels.

    Submodule attribute names (rgb_backbone / ir_backbone / depth_encoder)
    match train.py's backbone-vs-head parameter-group name matching.
    """

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        img_size = int(cfg["data"]["img_size"])
        self.d_model = int(mcfg["d_model"])
        tag = mcfg.get("tri_pretrained_tag", mcfg.get("swin_pretrained_tag", "ms_in22k"))
        pretrained = bool(mcfg.get("tri_pretrained", mcfg.get("swin_pretrained", True)))
        drop_path = float(mcfg.get("tri_drop_path_rate",
                                   mcfg.get("swin_drop_path_rate", 0.0)))

        def _path(key):
            v = mcfg.get(key)
            return v if v else None

        common = dict(pretrained=pretrained, pretrained_tag=tag,
                      img_size=img_size, drop_path_rate=drop_path)
        self.rgb_backbone = SwinModalityBackbone(
            variant=mcfg.get("tri_rgb_variant", "swin_tiny_patch4_window7_224"),
            in_chans=3,
            pretrained_path=_path("tri_rgb_pretrained_path"),
            freeze_stages=int(mcfg.get("tri_freeze_stages", 2)), **common)
        self.ir_backbone = SwinModalityBackbone(
            variant=mcfg.get("tri_ir_variant", "swin_tiny_patch4_window7_224"),
            in_chans=1,
            pretrained_path=_path("tri_ir_pretrained_path"),
            freeze_stages=int(mcfg.get("tri_freeze_stages", 2)), **common)
        # Depth is a geometric modality: freeze fewer stages so the 2ch-adapted
        # stem can adapt faster.
        self.depth_encoder = SwinModalityBackbone(
            variant=mcfg.get("tri_depth_variant", "swin_tiny_patch4_window7_224"),
            in_chans=2,
            pretrained_path=_path("tri_depth_pretrained_path"),
            freeze_stages=int(mcfg.get("tri_depth_freeze_stages", 1)), **common)

        embed_dim = self.rgb_backbone.embed_dim
        self._branches = (self.rgb_backbone, self.ir_backbone, self.depth_encoder)
        self.fuse_stage_idx = sorted(
            int(s) - 1 for s in mcfg.get("tri_fuse_stages", [2, 3, 4]))
        if not self.fuse_stage_idx:
            raise ValueError("tri_fuse_stages must contain at least one stage")
        window = int(mcfg.get("tri_window_size", 8))
        mdrop = float(mcfg.get("tri_modality_dropout", 0.1))
        self.fusers = nn.ModuleDict({
            str(i): HCMAFBlock(
                dim=embed_dim * (2 ** i),
                num_heads=embed_dim * (2 ** i) // 32,
                window_size=window, modality_dropout=mdrop)
            for i in self.fuse_stage_idx})
        fpn_in = [embed_dim * (2 ** i) for i in self.fuse_stage_idx]
        self.fpn = SimpleFPN(fpn_in, self.d_model)
        self.fpn_strides = [4 * (2 ** i) for i in self.fuse_stage_idx]
        self.encoder_strides = [int(s) for s in mcfg.get(
            "tri_encoder_scales", [16, 32])]
        unknown = set(self.encoder_strides) - set(self.fpn_strides)
        if unknown:
            raise ValueError(
                f"tri_encoder_scales {unknown} not in fused FPN strides "
                f"{self.fpn_strides}")

    @staticmethod
    def _pixel_mask_down(mask, hw):
        if mask is None:
            return None
        return F.interpolate(mask[None].float(), size=hw).to(torch.bool)[0]

    def forward(self, rgb, ir, depth, depth_mask, pixel_mask=None):
        depth_in = torch.cat(
            [depth, depth_mask.float().unsqueeze(1)], dim=1)  # (B,2,H,W)
        # All three towers run on NHWC maps up to the fusion points.
        maps = [bb.stem_maps(x)
                for bb, x in zip(self._branches, (rgb, ir, depth_in))]
        B = rgb.shape[0]
        H, W = rgb.shape[-2:]

        fused_by_stage = {}
        for i in range(4):
            maps = [bb.run_stage(i, m)
                    for bb, m in zip(self._branches, maps)]
            if i in self.fuse_stage_idx:
                h, w = H // (4 * 2 ** i), W // (4 * 2 ** i)
                nchw_maps = [SwinModalityBackbone.nhwc_to_nchw(m)
                             for m in maps]
                vis = self._pixel_mask_down(pixel_mask, (h, w))
                if vis is None:
                    vis = torch.ones((B, h, w), dtype=torch.bool,
                                     device=rgb.device)
                # Depth holes mask depth KEY tokens only; RGB/IR always valid.
                depth_vis = DepthEncoder.downsample_mask(depth_mask, (h, w)) & vis
                key_valid = [vis, vis, depth_vis]
                fused, z_maps, _ = self.fusers[str(i)](nchw_maps, key_valid)
                fused_by_stage[i] = fused
                # HCMAF outputs per-modality refined features: feed them back
                # into each tower's next stage.
                maps = [SwinModalityBackbone.nchw_to_nhwc(z)
                        for z in z_maps]

        feats = [fused_by_stage[i] for i in self.fuse_stage_idx]
        pyramids = self.fpn(feats)                    # high-res → low-res
        masks = [self._pixel_mask_down(pixel_mask, f.shape[-2:])
                 if pixel_mask is not None
                 else torch.ones((B, f.shape[2], f.shape[3]),
                                 dtype=torch.bool, device=f.device)
                 for f in pyramids]
        return pyramids, masks, self.fpn_strides


class TriModalSwinDETR(DetrForObjectDetection):
    """Three Swin towers + HCMAF + multi-scale FPN feeding the HF DETR head.

    Only P-scales listed in `tri_encoder_scales` are flattened into DETR
    encoder tokens; all FPN levels still interact via the top-down path, so
    e.g. P3's fine detail reaches the encoder through P4. Matcher, losses and
    heads are inherited unchanged; the forward signature mirrors
    MultiModalSwinDETR so train/eval/inference work unmodified.
    """

    def __init__(self, config: DetrConfig, cfg=None):
        super().__init__(config)
        self.loss_type = "ForObjectDetection"
        if cfg is None:
            raise ValueError("TriModalSwinDETR needs the run config `cfg`.")
        self.tri_backbone = TriModalSwinFusion(cfg)
        self.encoder_stride_set = set(self.tri_backbone.encoder_strides)
        self.scale_embed = nn.Embedding(
            len(self.tri_backbone.encoder_strides), config.d_model)
        nn.init.normal_(self.scale_embed.weight, std=0.02)
        # The HF DETR constructor built a throwaway timm resnet50 + 1x1
        # projection for DetrConvModel. We drive the encoder manually and
        # never call them; replace with Identity so their random-init
        # parameters don't enter the optimizer or the checkpoint. The sine
        # position_embedding (self.model.backbone.position_embedding) is
        # kept and reused in forward().
        self.model.backbone.conv_encoder = nn.Identity()
        self.model.input_projection = nn.Identity()
        # No post_init(): see MultiModalSwinDETR for why a second init would
        # destroy pretrained backbone weights.
        import math
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        with torch.no_grad():
            self.class_labels_classifier.bias.fill_(bias_value)

    def forward(
        self,
        pixel_values_rgb,
        pixel_values_ir,
        pixel_values_depth,
        depth_mask,
        pixel_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        device = pixel_values_rgb.device
        batch_size = pixel_values_rgb.shape[0]
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, pixel_values_rgb.shape[2], pixel_values_rgb.shape[3]),
                dtype=torch.bool, device=device)

        pyramids, fpn_masks, strides = self.tri_backbone(
            pixel_values_rgb, pixel_values_ir, pixel_values_depth,
            depth_mask, pixel_mask)

        embeds, attn_masks, poses = [], [], []
        scale_idx = 0
        # DETR's DetrConvModel normally computes per-map sine embeddings; its
        # position_embedding module is map-shape agnostic, so reuse it here.
        pos_embedder = self.model.backbone.position_embedding
        for fmap, mask, stride in zip(pyramids, fpn_masks, strides):
            if stride not in self.encoder_stride_set:
                continue
            pos = pos_embedder(fmap, mask)
            pos = pos + self.scale_embed.weight[scale_idx].view(1, -1, 1, 1)
            scale_idx += 1
            embeds.append(fmap.flatten(2).permute(0, 2, 1))
            attn_masks.append(mask.flatten(1))
            poses.append(pos.flatten(2).permute(0, 2, 1))
        flattened = torch.cat(embeds, dim=1)
        flattened_mask = torch.cat(attn_masks, dim=1)
        object_queries = torch.cat(poses, dim=1)

        encoder_outputs = self.model.encoder(
            inputs_embeds=flattened,
            attention_mask=flattened_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        query_pos = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size, 1, 1)
        queries = torch.zeros_like(query_pos)
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_pos,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                intermediate = (decoder_outputs.intermediate_hidden_states
                                if return_dict else decoder_outputs[4])
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, device, pred_boxes, self.config,
                outputs_class, outputs_coord)

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + decoder_outputs + encoder_outputs
            else:
                output = (logits, pred_boxes) + decoder_outputs + encoder_outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return DetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def freeze_backbone(self):
        """Freeze the three modality Swin towers (fusion + FPN + head train)."""
        for sub in (self.tri_backbone.rgb_backbone,
                    self.tri_backbone.ir_backbone,
                    self.tri_backbone.depth_encoder):
            for p in sub.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for sub in (self.tri_backbone.rgb_backbone,
                    self.tri_backbone.ir_backbone,
                    self.tri_backbone.depth_encoder):
            for p in sub.parameters():
                p.requires_grad = True


def build_detr_config(cfg):
    """Construct a DetrConfig from the run config.

    A throwaway `timm` resnet50 backbone is requested (use_timm_backbone=True,
    use_pretrained_backbone=False) just so DetrModel.__init__ can build a
    backbone whose `intermediate_channel_sizes[-1]` (2048) is consumed by
    `input_projection`; we immediately overwrite both `input_projection` and
    `conv_encoder` in MultiModalSwinDETR.__init__, so the throwaway backbone
    is discarded and never run.
    """
    mcfg = cfg["model"]
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    config = DetrConfig(
        num_labels=int(mcfg["num_labels"]),
        num_queries=int(mcfg["num_queries"]),
        d_model=int(mcfg["d_model"]),
        encoder_layers=int(mcfg["encoder_layers"]),
        decoder_layers=int(mcfg["decoder_layers"]),
        auxiliary_loss=bool(mcfg["auxiliary_loss"]),
        # Dropout rates (0 keeps train/eval forwards identical; small dataset).
        dropout=float(mcfg.get("detr_dropout", 0.1)),
        attention_dropout=float(mcfg.get("detr_attention_dropout", 0.0)),
        activation_dropout=float(mcfg.get("detr_activation_dropout", 0.0)),
        # Throwaway timm resnet50 — replaced in __init__.
        use_timm_backbone=True,
        backbone="resnet50",
        use_pretrained_backbone=False,
        id2label=id2label,
        label2id=label2id,
    )
    # DETR loss weights (defaults already match spec: ce=1, bbox=5, giou=2,
    # eos=0.1) but we surface them so the config is the single source of truth.
    loss = cfg.get("loss", {})
    if "loss_bbox" in loss:
        config.bbox_loss_coefficient = float(loss["loss_bbox"])
    if "loss_giou" in loss:
        config.giou_loss_coefficient = float(loss["loss_giou"])
    # class_cost / bbox_cost / giou_cost keep DETR defaults (1/5/2).
    return config


def build_model(cfg):
    """Factory: DetrConfig → detection model, selected by model.arch."""
    config = build_detr_config(cfg)
    arch = cfg["model"].get("arch", "cssa_detr")
    if arch == "tri_swin":
        return TriModalSwinDETR(config, cfg=cfg)
    if arch == "cssa_detr":
        return MultiModalSwinDETR(config, cfg=cfg)
    raise ValueError(f"Unknown model.arch: {arch!r} "
                     "(expected 'tri_swin' or 'cssa_detr')")

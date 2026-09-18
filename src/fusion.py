"""Cross-modal attention fusion modules.

- CSSAFusion (front-half, RGB ↔ IR): channel switching + spatial gating,
  following Cao et al. CVPRW 2023 "Multimodal object detection by channel
  switching and spatial attention". Switches low-scoring channels between
  modalities and gates the final mix per spatial location with a max+avg pool
  descriptor. Projects to DETR d_model.

- DepthLateFusion (back-half, depth injected): a single cross-attention layer
  placed between the multi-modal backbone output and DETR encoder. Fused
  RGB+IR features are Queries, depth features are Keys/Values; the depth
  validity mask is used as key_padding_mask. Leaves DETR's encoder/decoder
  untouched (zero-intrusion option C).
"""
import torch
import torch.nn as nn


class ChannelScorer(nn.Module):
    """SE-style channel attention: GAP → 1x1 reduction → ReLU → 1x1 expand → sigmoid."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.squeeze = nn.Conv1d(channels, hidden, 1, bias=True)
        self.excite = nn.Conv1d(hidden, channels, 1, bias=True)

    def forward(self, x):
        # x: (B, C, H, W) → GAP → (B, C, 1)
        z = x.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        z = z.squeeze(-1)                            # (B, C, 1) for Conv1d
        z = torch.relu(self.squeeze(z))
        z = torch.sigmoid(self.excite(z))            # (B, C, 1)
        return z.unsqueeze(-1)                       # (B, C, 1, 1)


class SpatialGate(nn.Module):
    """CBAM-style spatial gate: concat(avg+max pool across channels) → conv → sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)            # (B, 1, H, W)
        mx, _ = x.max(dim=1, keepdim=True)           # (B, 1, H, W)
        g = torch.cat([avg, mx], dim=1)              # (B, 2, H, W)
        return torch.sigmoid(self.conv(g))          # (B, 1, H, W)


class CSSAFusion(nn.Module):
    """Channel Switching Spatial Attention for RGB ↔ IR fusion.

    Inputs F_rgb and F_ir share channel count `channels` (the IRBackbone
    adapter projects to match Swin's output). Output is projected to
    `out_dim` (= DETR d_model).
    """

    def __init__(self, channels, out_dim, tau=0.5, reduction=16):
        super().__init__()
        self.tau = float(tau)  # fixed switching threshold; not learned initially
        self.rgb_scorer = ChannelScorer(channels, reduction)
        self.ir_scorer = ChannelScorer(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size=7)
        self.project = nn.Sequential(
            nn.Conv2d(channels, out_dim, 1, bias=False),
            nn.GroupNorm(8, out_dim),
        )

    def forward(self, f_rgb, f_ir):
        """
        f_rgb, f_ir: (B, channels, H, W)
        returns:    (B, out_dim, H, W)
        """
        # 1) Channel scores per modality.
        s_rgb = self.rgb_scorer(f_rgb)              # (B, C, 1, 1)
        s_ir = self.ir_scorer(f_ir)                 # (B, C, 1, 1)

        # 2) Channel switching: low-score channels borrow from the other modality.
        sw_rgb = torch.where(s_rgb < self.tau, f_ir, f_rgb)
        sw_ir = torch.where(s_ir < self.tau, f_rgb, f_ir)

        # 3) Spatial gate from the switched sum.
        g = self.spatial_gate(sw_rgb + sw_ir)       # (B, 1, H, W)

        # 4) Modality blending at each spatial location.
        fused = g * sw_rgb + (1.0 - g) * sw_ir      # (B, C, H, W)

        # 5) Project to DETR d_model.
        return self.project(fused)


class DepthLateFusion(nn.Module):
    """Single cross-attention layer injecting depth into the fused RGB+IR stream.

    Q = fused RGB+IR features; K=V = depth features. depth_mask (True=valid)
    is inverted to key_padding_mask (True=ignore) for nn.MultiheadAttention.
    Placed between the multi-modal backbone output and the DETR encoder.
    """

    def __init__(self, d_model=256, num_heads=8, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, fused, depth_feat, depth_mask=None):
        """
        fused:       (B, C, h, w)  (Q source)
        depth_feat:  (B, C, h, w)  (K/V source)
        depth_mask:  (B, h, w) bool, True where depth is valid; None = all valid
        returns:     (B, C, h, w)
        """
        B, C, h, w = fused.shape
        q = fused.flatten(2).transpose(1, 2)              # (B, hw, C)
        kv = depth_feat.flatten(2).transpose(1, 2)       # (B, hw, C)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        key_padding_mask = None
        if depth_mask is not None:
            # nn.MultiheadAttention: True in key_padding_mask == ignore that key.
            valid = depth_mask.flatten(1)                  # (B, hw) bool, True=valid
            key_padding_mask = ~valid                       # (B, hw), True=masked/invalid
            # Guard against fully-masked rows (would produce NaNs); keep at least
            # one valid key by un-masking the first valid pixel per sample.
            all_invalid = ~key_padding_mask.any(dim=1)
            if all_invalid.any():
                # fall back to unmasking the center key for those samples
                idx = (h * w) // 2
                key_padding_mask[all_invalid, idx] = False

        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask,
                                      need_weights=False)
        out = q + attn_out
        out = self.norm_out(out)
        out = out + self.ffn(out)

        return out.transpose(1, 2).reshape(B, C, h, w).contiguous()

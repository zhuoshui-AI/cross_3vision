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
import torch.nn.functional as F


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

    def __init__(self, d_model=256, num_heads=8, dropout=0.0, ffn_mult=4):
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


# ======================================================================
# TMSC-Det: Hierarchical Cross-Modal Attention Fusion (HCMAF)
# ======================================================================

def _partition_windows(x, ws):
    """(B, H, W, C) -> (B*nH*nW, ws*ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def _merge_windows(windows, ws, Hp, Wp):
    """Inverse of _partition_windows: -> (B, Hp, Wp, C)."""
    nH, nW = Hp // ws, Wp // ws
    B = windows.shape[0] // (nH * nW)
    C = windows.shape[-1]
    x = windows.view(B, nH, nW, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, Hp, Wp, C)


class JointWindowCrossAttention(nn.Module):
    """Window-based joint self-attention over ALL modalities at one scale.

    Inside each ws×ws window the tokens of all M=3 modalities participate in a
    single attention, so every spatial position simultaneously mixes space and
    modality. Q/K/V projections are shared across modalities (symmetric
    treatment, parameter-efficient). Attention bias = Swin-style spatial
    relative-position bias (shared across modality pairs) + a learned
    M×M modality-pair bias.

    Invalid depth tokens (and, during modality-dropout training, an entire
    dropped modality) are masked as keys. The output projection is zero-init
    so the block starts as an identity and never disturbs pretrained features.
    """

    def __init__(self, dim, num_heads, window_size=8, num_modal=3):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.ws = window_size
        self.M = num_modal
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        # Spatial relative-position bias (one table, shared across modality
        # pairs), indexed like vanilla Swin window attention.
        self.rpb_table = nn.Parameter(
            torch.zeros(num_heads, (2 * window_size - 1) ** 2))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size),
            indexing="ij"))  # (2, ws, ws)
        coords = coords.view(2, -1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel + window_size - 1
        index = (rel[0] * (2 * window_size - 1) + rel[1]).reshape(-1)
        self.register_buffer("rpb_index", index, persistent=False)
        # Modality-pair bias: bias[head, query_modality, key_modality].
        self.modal_bias = nn.Parameter(torch.zeros(num_heads, num_modal, num_modal))
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, key_valid=None, dropped_modality=None):
        """
        x: (M, B, C, H, W)
        key_valid: optional list of M (B, H, W) bool, True = valid token
        dropped_modality: optional int, this modality's keys are fully masked
        returns: (M, B, C, H, W)
        """
        M, B, C, H, W = x.shape
        ws, h, dh = self.ws, self.h, self.dh
        Hp = (H + ws - 1) // ws * ws
        Wp = (W + ws - 1) // ws * ws

        xc = x.permute(0, 1, 3, 4, 2)            # (M,B,H,W,C)
        xc = self.norm(xc)
        if Hp != H or Wp != W:
            xn = xc.permute(0, 1, 4, 2, 3)       # (M,B,C,H,W)
            xn = F.pad(xn, (0, Wp - W, 0, Hp - H))
            xc = xn.permute(0, 1, 3, 4, 2)

        nH, nW = Hp // ws, Wp // ws
        BnW = B * nH * nW
        N = ws * ws
        win = _partition_windows(
            xc.reshape(M * B, Hp, Wp, C), ws).view(M, BnW, N, C)

        q = self.q_proj(win).view(M, BnW, N, h, dh).permute(0, 1, 3, 2, 4)
        k = self.k_proj(win).view(M, BnW, N, h, dh).permute(1, 3, 0, 2, 4)
        v = self.v_proj(win).view(M, BnW, N, h, dh).permute(1, 3, 0, 2, 4)
        k = k.reshape(BnW, h, M * N, dh)         # columns: modality-major
        v = v.reshape(BnW, h, M * N, dh)

        attn = torch.einsum("mbhnd,bhjd->mbhnj", q, k) * (dh ** -0.5)
        attn = attn.view(M, BnW, h, N, M, N)

        spatial = self.rpb_table[:, self.rpb_index].view(h, N, N)
        # attn dims: (Mq, batch, head, q_spatial, Mk, k_spatial)
        attn = attn + spatial.view(1, 1, h, N, 1, N)
        # modal_bias is (h, Mq, Mk) -> align to attn dims (Mq,B,h,qN,Mk,kN).
        modal = self.modal_bias.permute(1, 0, 2).reshape(M, 1, h, 1, M, 1)
        attn = attn + modal

        if key_valid is not None:
            vals = []
            for m in range(M):
                msk = key_valid[m]
                if Hp != H or Wp != W:
                    msk = F.pad(msk, (0, Wp - W, 0, Hp - H),
                                mode="constant", value=False)
                vals.append(_partition_windows(
                    msk.unsqueeze(-1), ws).squeeze(-1))  # (BnW, N)
            valid = torch.cat(vals, dim=1)                # (BnW, M*N)
            attn = attn.masked_fill(
                ~valid.view(BnW, 1, 1, M, N), -1e4)
        if dropped_modality is not None:
            col_mask = torch.zeros(M, N, dtype=torch.bool, device=x.device)
            col_mask[dropped_modality, :] = True
            attn = attn.masked_fill(col_mask.view(1, 1, 1, M, N), -1e4)

        attn = attn.view(M, BnW, h, N, M * N).softmax(dim=-1)
        out = torch.einsum("mbhnj,bhjd->mbhnd", attn, v)  # (M,BnW,h,N,dh)
        out = out.permute(0, 1, 3, 2, 4).reshape(M * BnW, N, C)
        out = self.proj(out)

        out = _merge_windows(out, ws, Hp, Wp)             # (M*B,Hp,Wp,C)
        out = out.view(M, B, Hp, Wp, C)[:, :, :H, :W, :]  # crop padding
        out = out.permute(0, 1, 4, 2, 3).contiguous()     # (M,B,C,H,W)
        return x + out


class HCMAFBlock(nn.Module):
    """One fusion block at a single scale.

    Pipeline per scale: shared joint-window cross-attention across modalities
    → per-modality FFN residual → reliability-gated aggregation into one fused
    map. Training-only modality dropout masks one whole modality (attention
    keys + gate) so the fused prediction stays usable when a sensor degrades.
    """

    def __init__(self, dim, num_heads, window_size=8, num_modal=3,
                 ffn_mult=4, gate_reduction=4, dropout=0.0,
                 modality_dropout=0.1):
        super().__init__()
        self.M = num_modal
        self.modality_dropout = float(modality_dropout)
        self.attn = JointWindowCrossAttention(
            dim, num_heads, window_size=window_size, num_modal=num_modal)
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_modal)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ffn_mult, dim),
                nn.Dropout(dropout),
            ) for _ in range(num_modal)])
        for net in self.ffn:
            nn.init.zeros_(net[-2].weight)
            nn.init.zeros_(net[-2].bias)
        hidden = max(dim // gate_reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feats, key_valid=None):
        """
        feats: list of M maps (B, C, H, W)
        key_valid: optional list of M (B, H, W) bool
        returns: fused (B,C,H,W), updated list of M maps, alpha (B,M)
        """
        x = torch.stack(feats, dim=0)                 # (M,B,C,H,W)
        dropped = None
        if self.training and self.modality_dropout > 0 \
                and torch.rand(1, device=x.device).item() < self.modality_dropout:
            dropped = int(torch.randint(0, self.M, (1,), device=x.device).item())

        attn_out = self.attn(x, key_valid=key_valid, dropped_modality=dropped)
        z = x + attn_out                              # (M,B,C,H,W)

        z_maps = []
        for m in range(self.M):
            zm = z[m].permute(0, 2, 3, 1)            # (B,H,W,C)
            zm = zm + self.ffn[m](self.ffn_norm[m](zm))
            z_maps.append(zm.permute(0, 3, 1, 2).contiguous())

        # Reliability gate: one scalar per (sample, modality), softmax over M.
        z_stack = torch.stack(z_maps, dim=0)          # (M,B,C,H,W)
        g = self.gate(z_stack.mean(dim=(-2, -1)))     # (M,B,1)
        logits = g.squeeze(-1).permute(1, 0)          # (B,M)
        if dropped is not None:
            logits = logits.clone()
            logits[:, dropped] = float("-inf")
        alpha = logits.softmax(dim=1)                 # (B,M)
        fused = sum(alpha[:, m].view(-1, 1, 1, 1) * z_maps[m]
                    for m in range(self.M))
        return fused, z_maps, alpha


class SimpleFPN(nn.Module):
    """FPN top-down + PAN bottom-up, all scales -> out_dim.

    Input list is ordered high-resolution → low-resolution (e.g. stride
    8/16/32). The top-down path spreads semantics P5→P3; the PAN bottom-up
    path then pushes P3's fine detail back into P4/P5. This matters because
    the DETR encoder may consume only P4/P5: without the bottom-up leg, P3
    (and the HCMAF gate producing it) would be a dead branch receiving no
    gradient.
    """

    def __init__(self, in_channels, out_dim, groups=8):
        super().__init__()

        def _gn(c):
            gg = groups
            while c % gg != 0 and gg > 1:
                gg -= 1
            return nn.GroupNorm(gg, c)

        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_dim, 1, bias=False), _gn(out_dim))
            for c in in_channels])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                          _gn(out_dim))
            for _ in in_channels])
        # PAN bottom-up merges: stride-2 conv P_i -> P_{i+1}.
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1,
                          bias=False),
                _gn(out_dim))
            for _ in range(len(in_channels) - 1)])

    def forward(self, feats):
        lats = [l(f) for l, f in zip(self.lateral, feats)]
        topdown = [None] * len(lats)
        topdown[-1] = lats[-1]
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(topdown[i + 1], size=lats[i].shape[-2:],
                               mode="nearest")
            topdown[i] = lats[i] + up
        td = [s(o) for s, o in zip(self.smooth, topdown)]
        # Bottom-up PAN: each coarser level adds the up-projected finer one.
        outs = [td[0]]
        for i in range(len(td) - 1):
            d = self.down[i](outs[-1])
            # PatchMerging pads odd sizes; match via interpolation if needed.
            if d.shape[-2:] != td[i + 1].shape[-2:]:
                d = F.interpolate(d, size=td[i + 1].shape[-2:],
                                  mode="nearest")
            outs.append(td[i + 1] + d)
        return outs

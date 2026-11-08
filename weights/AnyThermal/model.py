import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, Dinov2Model, Dinov2Config

# =============================================================================
# HELPER: VPR Sinkhorn (Matches salad.py)
# =============================================================================
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    M = M / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm

# =============================================================================
# 1. SEGMENTATION MODEL
# Matches NonLinearSegmentationHead64: Conv(0)->ReLU(1)->Dropout(2)->Conv(3)
# =============================================================================
class AnyThermalConfig(Dinov2Config):
    model_type = "anythermal"

class AnyThermalSegmentationModel(PreTrainedModel):
    config_class = AnyThermalConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = Dinov2Model(config)
        
        # Head definition matches your NonlinearHead64
        self.head = nn.Module()
        self.head.model = nn.Sequential(
            nn.Conv2d(config.hidden_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.0), 
            nn.Conv2d(64, config.num_labels, kernel_size=1)
        )
        
        # Define Normalization constants as buffers so they move to GPU automatically
        self.register_buffer("norm_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        
        self.post_init()

    def preprocess_input(self, x):
        """
        Replicates preprocess_dinov2: 
        1. Resize to nearest multiple of 14
        2. Normalize with ViT stats
        """
        B, C, H, W = x.shape
        patch_size = 14
        
        # 1. Dynamic Resize (Snap to grid)
        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size
        
        if new_H != H or new_W != W:
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            
        # 2. Normalize

        if x.max() > 1.0: x = x / 255.0
        
        x = (x - self.norm_mean) / self.norm_std
        return x

    def forward(self, pixel_values, labels=None, **kwargs):
        # --- APPLY PREPROCESSING HERE ---
        pixel_values = self.preprocess_input(pixel_values)
        # --------------------------------
        
        outputs = self.backbone(pixel_values, **kwargs)
        features = outputs.last_hidden_state[:, 1:, :] 
        B, L, C = features.shape
        H = W = int(L**0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        logits = self.head.model(features)
        
        # Upscale back to input size
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
            
        return logits


# =============================================================================
# 2. VPR MODEL (SALAD)
# Matches salad.py: Conv(0)->Dropout(1)->ReLU(2)->Conv(3) + dust_bin
# =============================================================================
class AnyThermalVPRConfig(Dinov2Config):
    model_type = "anythermal_vpr"
    def __init__(self, num_clusters=64, cluster_dim=128, token_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

class SALADHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = config.hidden_size
        self.num_clusters = config.num_clusters
        self.cluster_dim = config.cluster_dim
        self.token_dim = config.token_dim

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        
        # Matches salad.py structure
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            nn.Dropout(0.0), 
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            nn.Dropout(0.0),
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x_tuple):
        x, t = x_tuple 
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :] 

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f_rep = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        vlad = (f_rep * p).sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=1).flatten(1)
        
        combined = torch.cat([F.normalize(t, p=2, dim=-1), vlad], dim=-1)
        return F.normalize(combined, p=2, dim=-1)

class AnyThermalVPRModel(PreTrainedModel):
    config_class = AnyThermalVPRConfig
    def __init__(self, config):
        super().__init__(config)
        self.backbone = Dinov2Model(config)
        # Sequential wrapper to match checkpoint key "0.cluster_features"
        self.vpr_head = nn.Sequential(SALADHead(config))
        self.post_init()

    def forward(self, pixel_values, **kwargs):
        outputs = self.backbone(pixel_values, **kwargs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :].permute(0, 2, 1)
        B, C, L = patch_tokens.shape
        H = W = int(L**0.5)
        patch_tokens = patch_tokens.reshape(B, C, H, W)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.vpr_head[0]((patch_tokens, cls_token))


# =============================================================================
# 3. DEPTH MODEL (MiDaS)
# Matches vit.py indices: Identity(0,1,2) -> Conv(3) -> ConvTranspose(4)
# =============================================================================
class AnyThermalDepthConfig(Dinov2Config):
    model_type = "anythermal_depth"
    def __init__(self, features=256, **kwargs):
        super().__init__(**kwargs)
        self.features = features

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            if output.shape[-2:] != xs[1].shape[-2:]:
                output = F.interpolate(output, size=xs[1].shape[-2:], mode="bilinear", align_corners=True)
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output

class AnyThermalDepthModel(PreTrainedModel):
    config_class = AnyThermalDepthConfig
    def __init__(self, config):
        super().__init__(config)
        self.backbone = Dinov2Model(config)
        features = config.features
        
        self.scratch = nn.Module()
        self.pretrained = nn.Module()

        self.scratch.layer1_rn = nn.Conv2d(96, features, 3, 1, 1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(192, features, 3, 1, 1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(384, features, 3, 1, 1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(768, features, 3, 1, 1, bias=False)
        
        # Padded with 3 Identities to shift Conv indices to 3 and 4
        # This aligns with the checkpoint keys (which had Slice/Transpose/Unflatten at 0-2)
        self.pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv2d(768, 96, 1), nn.ConvTranspose2d(96, 96, 4, 4)
        )
        self.pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv2d(768, 192, 1), nn.ConvTranspose2d(192, 192, 2, 2)
        )
        self.pretrained.act_postprocess3 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv2d(768, 384, 1)
        )
        self.pretrained.act_postprocess4 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv2d(768, 768, 1), nn.Conv2d(768, 768, 3, 2, 1)
        )
        
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, 3, 1, 1),
            nn.Upsample(scale_factor=1.75, mode="bilinear"), 
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.ReLU(True)
        )
        self.post_init()

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        layers = [outputs.hidden_states[i] for i in [3, 6, 9, 12]]
        
        def process(l, h, w):
            l = l[:, 1:, :].transpose(1, 2)
            return l.reshape(l.shape[0], l.shape[1], h//14, w//14)
            
        b, _, h, w = pixel_values.shape
        l1, l2, l3, l4 = [process(layers[i], h, w) for i in range(4)]
        
        layer_1_rn = self.scratch.layer1_rn(self.pretrained.act_postprocess1(l1))
        layer_2_rn = self.scratch.layer2_rn(self.pretrained.act_postprocess2(l2))
        layer_3_rn = self.scratch.layer3_rn(self.pretrained.act_postprocess3(l3))
        layer_4_rn = self.scratch.layer4_rn(self.pretrained.act_postprocess4(l4))

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return self.scratch.output_conv(path_1).squeeze(1)

# Register all classes
AnyThermalSegmentationModel.register_for_auto_class("AutoModel")
AnyThermalVPRModel.register_for_auto_class("AutoModel")
AnyThermalDepthModel.register_for_auto_class("AutoModel")
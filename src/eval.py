"""Validation mAP@50-95 using torchmetrics.

DETR outputs: logits (B, Q, num_labels+1), pred_boxes (B, Q, 4) cxcywh
normalized. We softmax over the last dim, drop the "no-object" class (index
-1), keep the per-query max class/score, filter by conf_threshold, scale boxes
to pixel coords, and feed torchmetrics MeanAveragePrecision.
"""
import torch

try:
    from torchmetrics.detection import MeanAveragePrecision
    _HAS_TM = True
except Exception:  # pragma: no cover
    _HAS_TM = False

from .losses import cxcywh_to_xyxy_pixel


@torch.no_grad()
def evaluate(model, val_loader, device, cfg, amp_enabled=False):
    """Returns dict with mAP@50-95 and per-class AP if torchmetrics present."""
    if not _HAS_TM:
        return {"map_5095": float("nan"), "note": "torchmetrics unavailable"}
    model.eval()
    num_labels = int(cfg["model"]["num_labels"])
    conf_th = float(cfg["inference"]["conf_threshold"])
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox",
        iou_thresholds=[round(v, 2) for v in
                        torch.linspace(0.5, 0.95, 10).tolist()],
        class_metrics=True,
    )
    for batch in val_loader:
        rgb = batch["pixel_values_rgb"].to(device)
        ir = batch["pixel_values_ir"].to(device)
        depth = batch["pixel_values_depth"].to(device)
        depth_mask = batch["depth_mask"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]
        img_size = (rgb.shape[2], rgb.shape[3])

        out = model(pixel_values_rgb=rgb, pixel_values_ir=ir,
                    pixel_values_depth=depth, depth_mask=depth_mask,
                    pixel_mask=pixel_mask, labels=None)
        probs = out.logits.softmax(-1)            # (B, Q, C+1)
        scores, labels_pred = probs[..., :-1].max(-1)  # drop no-object
        boxes_cxcywh = out.pred_boxes              # (B, Q, 4) cxcywh norm

        preds, targets = [], []
        for b in range(rgb.shape[0]):
            keep = scores[b] >= conf_th
            bx = cxcywh_to_xyxy_pixel(boxes_cxcywh[b][keep], img_size).cpu()
            preds.append({
                "boxes": bx,
                "scores": scores[b][keep].cpu(),
                "labels": labels_pred[b][keep].cpu(),
            })
            tgt = labels[b]
            tgt_boxes = cxcywh_to_xyxy_pixel(tgt["boxes"], img_size).cpu()
            targets.append({
                "boxes": tgt_boxes,
                "labels": tgt["class_labels"].cpu(),
            })
        metric.update(preds, targets)

    res = metric.compute()
    return {
        "map_5095": float(res["map"].item()),
        "map_50": float(res["map_50"].item()),
        "mar_100": float(res["mar_100"].item()),
        "per_class_ap": res.get("map_per_class", None),
    }

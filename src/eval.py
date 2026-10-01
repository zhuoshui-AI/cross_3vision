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
def evaluate(model, val_loader, device, cfg, amp_enabled=False, conf_th=0.0):
    """Returns dict with mAP@50-95 and per-class AP if torchmetrics present.

    conf_th defaults to 0.0 = keep ALL queries: pre-filtering low-score
    predictions drops true positives and systematically under-reports mAP
    (DETR convention is to feed every query to the metric). Thresholding
    belongs to submission-time output (inference.py), not evaluation.
    """
    if not _HAS_TM:
        return {"map_5095": float("nan"), "note": "torchmetrics unavailable"}
    model.eval()
    num_labels = int(cfg["model"]["num_labels"])
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

    def _f(key):
        # 数据集目标过小（无 medium/large 框）时 torchmetrics 返回 -1 或缺失，
        # DETR log.txt 约定用总指标填充这些槽位。
        try:
            v = float(res[key].item())
        except (KeyError, AttributeError):
            v = -1.0
        return v

    map_all, mar_all = _f("map"), _f("mar_100")
    return {
        "map_5095": map_all,
        "map_50": _f("map_50"),
        "map_75": _f("map_75"),
        # 无 s/m/l 分层时退化为总指标，保持 test_coco_eval_bbox 12 项齐全
        "map_small": _f("map_small") if _f("map_small") >= 0 else map_all,
        "map_medium": _f("map_medium") if _f("map_medium") >= 0 else map_all,
        "map_large": _f("map_large") if _f("map_large") >= 0 else map_all,
        "mar_1": _f("mar_1"),
        "mar_10": _f("mar_10"),
        "mar_100": mar_all,
        "mar_small": _f("mar_small") if _f("mar_small") >= 0 else mar_all,
        "mar_medium": _f("mar_medium") if _f("mar_medium") >= 0 else mar_all,
        "mar_large": _f("mar_large") if _f("mar_large") >= 0 else mar_all,
        "per_class_ap": res.get("map_per_class", None),
    }

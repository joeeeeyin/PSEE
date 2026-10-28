import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tad.temporal_detr import (
    TADDataset,
    WearSecondEncoder,
    collate_tad,
    evaluate_map,
    load_wear_records,
    load_wifi_records,
    predict,
    segment_iou_np,
)


def robust_scale(values):
    low = torch.quantile(values, 0.1)
    high = torch.quantile(values, 0.95)
    return ((values - low) / (high - low + 1e-6)).clamp(0, 1)


def transition_prior(feats, modality):
    if modality == "wear":
        steps = feats.reshape(feats.shape[0], 12, 50)
        signature = torch.cat([steps.mean(-1), steps.std(-1)], dim=-1)
    else:
        signature = feats
    diff = (signature[1:] - signature[:-1]).abs()
    energy = diff.mean(-1)
    channel_threshold = diff.median(dim=-1, keepdim=True).values
    consistency = (diff >= channel_threshold).float().mean(-1)
    score = energy * (0.5 + consistency)
    score = torch.cat([score[:1], score], dim=0) if len(score) else torch.zeros(1)
    return robust_scale(score)


def point_annotations(record):
    duration = max(float(record["duration"]), 1e-6)
    return [
        {
            "time": 0.5 * (float(ann["segment"][0]) + float(ann["segment"][1])),
            "position": 0.5
            * (float(ann["segment"][0]) + float(ann["segment"][1]))
            / duration,
            "label": int(ann["label"]),
            "annotation_index": idx,
        }
        for idx, ann in enumerate(record["annotations"])
    ]


class PointWindowDataset(Dataset):
    def __init__(self, records, modality, window_length, stride, max_items=None):
        self.records = records[:max_items] if max_items else records
        self.modality = modality
        self.samples = []
        for record_idx, record in enumerate(self.records):
            length = int(np.load(record["feat_path"], mmap_mode="r").shape[0])
            points = point_annotations(record)
            duration = max(float(record["duration"]), 1e-6)
            points = [
                {
                    "step": p["time"] / duration * length,
                    "label": p["label"],
                }
                for p in points
            ]
            if length <= window_length:
                starts = [0]
            else:
                starts = list(range(0, length - window_length + 1, stride))
                if starts[-1] != length - window_length:
                    starts.append(length - window_length)
            for start in starts:
                end = min(start + window_length, length)
                local_points = [
                    {"step": p["step"] - start, "label": p["label"]}
                    for p in points
                    if start <= p["step"] < end
                ]
                self.samples.append((record_idx, start, end, local_points))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_idx, start, end, points = self.samples[idx]
        record = self.records[record_idx]
        feats = np.load(record["feat_path"], mmap_mode="r")[start:end].astype(np.float32)
        feats = np.nan_to_num(feats.reshape(feats.shape[0], -1))
        feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-6)
        feats = torch.from_numpy(feats)
        return {
            "feats": feats,
            "prior": transition_prior(feats, self.modality),
            "points": points,
        }


def collate_points(batch):
    max_len = max(len(item["feats"]) for item in batch)
    dims = batch[0]["feats"].shape[-1]
    feats = torch.zeros(len(batch), max_len, dims)
    prior = torch.zeros(len(batch), max_len)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    points = []
    lengths = []
    for i, item in enumerate(batch):
        length = len(item["feats"])
        feats[i, :length] = item["feats"]
        prior[i, :length] = item["prior"]
        mask[i, :length] = False
        points.append(item["points"])
        lengths.append(length)
    return feats, prior, mask, points, lengths


class DilatedResidual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)


class WeakPointLocalizer(nn.Module):
    def __init__(self, input_dim, num_classes, modality, hidden_dim=128, downsample=8):
        super().__init__()
        self.modality = modality
        self.second_encoder = WearSecondEncoder(128) if modality == "wear" else None
        in_dim = 128 if modality == "wear" else input_dim
        blocks = []
        stride_left = downsample
        while stride_left > 1:
            blocks.extend(
                [
                    nn.Conv1d(in_dim, hidden_dim, 5, stride=2, padding=2),
                    nn.GroupNorm(8, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_dim = hidden_dim
            stride_left //= 2
        blocks.extend(
            [
                DilatedResidual1D(hidden_dim, 1),
                DilatedResidual1D(hidden_dim, 2),
                DilatedResidual1D(hidden_dim, 4),
            ]
        )
        self.encoder = nn.Sequential(*blocks)
        self.semantic_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.transition_head = nn.Conv1d(hidden_dim, 1, 1)
        nn.init.constant_(self.semantic_head.bias, -2.0)

    def forward(self, feats, mask):
        if self.second_encoder is not None:
            feats = self.second_encoder(feats)
        encoded = self.encoder(feats.transpose(1, 2))
        out_mask = F.interpolate(mask[:, None].float(), encoded.shape[-1], mode="nearest")[:, 0].bool()
        return {
            "cam_logits": self.semantic_head(encoded),
            "transition_logits": self.transition_head(encoded)[:, 0],
            "mask": out_mask,
        }


def localizer_loss(outputs, prior, points, lengths, num_classes):
    cam_logits = outputs["cam_logits"]
    transition_logits = outputs["transition_logits"]
    valid = ~outputs["mask"]
    out_len = cam_logits.shape[-1]
    point_ce = cam_logits.sum() * 0
    point_pos = cam_logits.sum() * 0
    point_count = 0
    sequence_targets = torch.zeros(cam_logits.shape[0], num_classes, device=cam_logits.device)
    for batch_idx, sample_points in enumerate(points):
        for point in sample_points:
            out_idx = min(int(point["step"] / max(lengths[batch_idx], 1) * out_len), out_len - 1)
            label = int(point["label"])
            logits = cam_logits[batch_idx, :, out_idx][None]
            point_ce = point_ce + F.cross_entropy(logits, torch.tensor([label], device=logits.device))
            point_pos = point_pos + F.binary_cross_entropy_with_logits(
                cam_logits[batch_idx, label, out_idx],
                torch.ones((), device=logits.device),
            )
            sequence_targets[batch_idx, label] = 1
            point_count += 1
    point_ce = point_ce / max(point_count, 1)
    point_pos = point_pos / max(point_count, 1)

    masked_logits = cam_logits.masked_fill(~valid[:, None], -20)
    topk = max(1, math.ceil(out_len * 0.05))
    sequence_logits = masked_logits.topk(topk, dim=-1).values.mean(-1)
    mil_loss = F.binary_cross_entropy_with_logits(sequence_logits, sequence_targets)

    prior_out = F.interpolate(prior[:, None], out_len, mode="linear", align_corners=False)[:, 0]
    transition_loss = F.binary_cross_entropy_with_logits(
        transition_logits[valid],
        prior_out[valid],
    )
    cam_prob = cam_logits.sigmoid()
    smooth_weight = 1 - torch.maximum(prior_out[:, 1:], prior_out[:, :-1])
    smooth = (
        (cam_prob[:, :, 1:] - cam_prob[:, :, :-1]).abs()
        * smooth_weight[:, None]
        * valid[:, None, 1:]
    ).sum() / valid[:, 1:].sum().clamp(min=1)
    sparsity = (cam_prob * valid[:, None]).sum() / valid.sum().clamp(min=1)
    total = point_ce + 0.5 * point_pos + 0.5 * mil_loss + 0.2 * transition_loss + 0.05 * smooth + 0.002 * sparsity
    return {
        "loss": total,
        "point_ce": point_ce,
        "mil": mil_loss,
        "transition": transition_loss,
        "smooth": smooth,
    }


def train_localizer(args, records, labels, modality, output_dir):
    window = args.wifi_window if modality == "wifi" else args.wear_window
    stride = args.wifi_stride if modality == "wifi" else args.wear_stride
    dataset = PointWindowDataset(
        records,
        modality,
        window,
        stride,
        max_items=args.max_train_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.localizer_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_points,
    )
    downsample = args.wifi_localizer_downsample if modality == "wifi" else args.wear_localizer_downsample
    model = WeakPointLocalizer(
        30 if modality == "wifi" else 600,
        len(labels),
        modality,
        hidden_dim=args.localizer_hidden_dim,
        downsample=downsample,
    ).to(args.device)
    if args.localizer_checkpoint:
        model.load_state_dict(torch.load(args.localizer_checkpoint, map_location=args.device))
        print(f"weak-localizer reused={args.localizer_checkpoint}", flush=True)
        return model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.localizer_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.localizer_epochs, eta_min=1e-6)
    print(f"weak-localizer windows={len(dataset)} point_labels_only=1", flush=True)
    for epoch in range(1, args.localizer_epochs + 1):
        model.train()
        running = defaultdict(float)
        for feats, prior, mask, points, lengths in loader:
            feats, prior, mask = feats.to(args.device), prior.to(args.device), mask.to(args.device)
            outputs = model(feats, mask)
            losses = localizer_loss(outputs, prior, points, lengths, len(labels))
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                running[key] += float(value.detach())
        scheduler.step()
        denom = max(len(loader), 1)
        print(
            f"weak_epoch={epoch:03d} "
            + " ".join(f"{key}={value / denom:.4f}" for key, value in sorted(running.items())),
            flush=True,
        )
    torch.save(model.state_dict(), output_dir / "weak_localizer.pt")
    return model


@torch.no_grad()
def infer_weak_maps(model, record, modality, device):
    feats = np.load(record["feat_path"]).astype(np.float32)
    feats = np.nan_to_num(feats.reshape(feats.shape[0], -1))
    feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-6)
    feats = torch.from_numpy(feats)
    prior = transition_prior(feats, modality)
    outputs = model(
        feats[None].to(device),
        torch.zeros(1, len(feats), dtype=torch.bool, device=device),
    )
    cam = outputs["cam_logits"].sigmoid()[0].cpu()
    learned_transition = outputs["transition_logits"].sigmoid()[0].cpu()
    prior_out = F.interpolate(prior[None, None], cam.shape[-1], mode="linear", align_corners=False)[0, 0]
    transition = 0.5 * learned_transition + 0.5 * prior_out
    cam = F.avg_pool1d(cam[None], 5, stride=1, padding=2)[0]
    return cam, transition


def expand_from_point(signal, point, left_limit, right_limit):
    point = int(max(left_limit, min(point, right_limit - 1)))
    peak = max(float(signal[point]), 0.05)
    local = signal[left_limit:right_limit]
    threshold = max(0.15, 0.35 * peak, float(torch.quantile(local, 0.4)) if len(local) else 0.15)
    left = point
    misses = 0
    while left > left_limit:
        left -= 1
        misses = misses + 1 if signal[left] < threshold else 0
        if misses >= 2:
            left += 2
            break
    right = point + 1
    misses = 0
    while right < right_limit:
        misses = misses + 1 if signal[right] < threshold else 0
        right += 1
        if misses >= 2:
            right -= 2
            break
    return max(left_limit, left), min(right_limit, max(right, point + 1))


def snap_boundary(transition, boundary, low, high):
    low = max(int(low), 0)
    high = min(int(high), len(transition))
    if high <= low:
        return int(boundary)
    return low + int(torch.argmax(transition[low:high]))


def generate_pseudo_records(model, records, modality, device, boundary_fraction=0.0, seed=2026):
    pseudo_records = []
    quality = []
    weak_quality = []
    boundary_count = 0
    rng = np.random.RandomState(seed)
    annotation_keys = [
        (record_idx, annotation_idx)
        for record_idx, record in enumerate(records)
        for annotation_idx in range(len(record["annotations"]))
    ]
    permutation = rng.permutation(len(annotation_keys))
    selected_count = int(round(boundary_fraction * len(annotation_keys)))
    if boundary_fraction > 0 and selected_count == 0 and annotation_keys:
        selected_count = 1
    selected_boundaries = {
        annotation_keys[index] for index in permutation[:selected_count]
    }
    for record_idx, record in enumerate(records):
        cam, transition = infer_weak_maps(model, record, modality, device)
        points = point_annotations(record)
        points.sort(key=lambda item: item["position"])
        pseudo_annotations = []
        for idx, point in enumerate(points):
            position = point["position"] * len(cam)
            left_limit = 0 if idx == 0 else int(0.5 * (points[idx - 1]["position"] + point["position"]) * len(cam))
            right_limit = len(cam) if idx == len(points) - 1 else int(
                0.5 * (point["position"] + points[idx + 1]["position"]) * len(cam)
            )
            left, right = expand_from_point(
                cam[point["label"]],
                int(position),
                left_limit,
                max(right_limit, left_limit + 1),
            )
            radius = max(3, int(0.12 * max(right_limit - left_limit, 1)))
            left = snap_boundary(transition, left, max(left_limit, left - radius), min(int(position), left + radius + 1))
            right_peak = snap_boundary(
                transition,
                right - 1,
                max(int(position) + 1, right - radius),
                min(right_limit, right + radius + 1),
            )
            right = max(right_peak + 1, int(position) + 1)
            duration = float(record["duration"])
            segment = [left / len(cam) * duration, right / len(cam) * duration]
            annotation_idx = point["annotation_index"]
            uses_boundary = (record_idx, annotation_idx) in selected_boundaries
            if uses_boundary:
                segment = list(record["annotations"][annotation_idx]["segment"])
                boundary_count += 1
            pseudo_annotations.append({"segment": segment, "label": point["label"]})
            gt = record["annotations"][annotation_idx]["segment"]
            quality.append(segment_iou_np(segment, gt))
            if not uses_boundary:
                weak_quality.append(quality[-1])
        pseudo = copy.deepcopy(record)
        pseudo["annotations"] = pseudo_annotations
        pseudo_records.append(pseudo)
    return pseudo_records, quality, weak_quality, boundary_count


class DenseTemporalDetector(nn.Module):
    def __init__(self, input_dim, num_classes, modality, hidden_dim=192, downsample=16):
        super().__init__()
        self.second_encoder = WearSecondEncoder(128) if modality == "wear" else None
        in_dim = 128 if modality == "wear" else input_dim
        blocks = []
        stride_left = downsample
        while stride_left > 1:
            blocks.extend(
                [
                    nn.Conv1d(in_dim, hidden_dim, 5, stride=2, padding=2),
                    nn.GroupNorm(8, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_dim = hidden_dim
            stride_left //= 2
        blocks.extend(
            [
                DilatedResidual1D(hidden_dim, 1),
                DilatedResidual1D(hidden_dim, 2),
                DilatedResidual1D(hidden_dim, 4),
            ]
        )
        self.encoder = nn.Sequential(*blocks)
        self.class_head = self._head(hidden_dim, num_classes)
        self.reg_head = self._head(hidden_dim, 2)
        self.center_head = self._head(hidden_dim, 1)
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - 0.01) / 0.01))

    @staticmethod
    def _head(hidden_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_dim, 3, padding=1),
        )

    def forward(self, feats, mask):
        if self.second_encoder is not None:
            feats = self.second_encoder(feats)
        encoded = self.encoder(feats.transpose(1, 2))
        out_mask = F.interpolate(mask[:, None].float(), encoded.shape[-1], mode="nearest")[:, 0].bool()
        return {
            "pred_logits": self.class_head(encoded),
            "pred_offsets": self.reg_head(encoded).sigmoid(),
            "pred_centerness": self.center_head(encoded)[:, 0],
            "mask": out_mask,
        }


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * ((1 - p_t) ** gamma) * ce


def dense_detector_loss(outputs, targets, num_classes):
    logits = outputs["pred_logits"]
    offsets = outputs["pred_offsets"]
    centerness = outputs["pred_centerness"]
    valid = ~outputs["mask"]
    batch_size, _, length = logits.shape
    locations = (torch.arange(length, device=logits.device).float() + 0.5) / length
    cls_targets = torch.zeros_like(logits)
    reg_targets = torch.zeros_like(offsets)
    center_targets = torch.zeros_like(centerness)
    positive = torch.zeros_like(valid)
    for batch_idx, target in enumerate(targets):
        if not len(target["boxes"]):
            continue
        boxes = target["boxes"].to(logits.device)
        labels = target["labels"].to(logits.device)
        starts = boxes[:, 0] - 0.5 * boxes[:, 1]
        ends = boxes[:, 0] + 0.5 * boxes[:, 1]
        left = locations[:, None] - starts[None]
        right = ends[None] - locations[:, None]
        inside = (left > 0) & (right > 0)
        widths = boxes[:, 1][None].expand(length, -1).masked_fill(~inside, float("inf"))
        best_width, best_idx = widths.min(dim=1)
        pos = torch.isfinite(best_width) & valid[batch_idx]
        if not pos.any():
            continue
        matched = best_idx[pos]
        positive[batch_idx, pos] = True
        cls_targets[batch_idx, labels[matched], pos] = 1
        matched_left = left[pos, matched]
        matched_right = right[pos, matched]
        reg_targets[batch_idx, 0, pos] = matched_left
        reg_targets[batch_idx, 1, pos] = matched_right
        center_targets[batch_idx, pos] = torch.sqrt(
            torch.minimum(matched_left, matched_right)
            / torch.maximum(matched_left, matched_right).clamp(min=1e-6)
        )

    num_pos = positive.sum().clamp(min=1)
    cls_loss = (sigmoid_focal_loss(logits, cls_targets) * valid[:, None]).sum() / num_pos
    if positive.any():
        pred_reg = offsets.permute(0, 2, 1)[positive]
        target_reg = reg_targets.permute(0, 2, 1)[positive]
        reg_loss = F.smooth_l1_loss(pred_reg, target_reg, reduction="sum") / num_pos
        pred_start = -pred_reg[:, 0]
        pred_end = pred_reg[:, 1]
        target_start = -target_reg[:, 0]
        target_end = target_reg[:, 1]
        inter = torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)
        union = torch.maximum(pred_end, target_end) - torch.minimum(pred_start, target_start)
        iou_loss = (1 - inter.clamp(min=0) / union.clamp(min=1e-6)).sum() / num_pos
        center_loss = F.binary_cross_entropy_with_logits(
            centerness[positive],
            center_targets[positive],
            reduction="sum",
        ) / num_pos
    else:
        zero = offsets.sum() * 0
        reg_loss = zero
        iou_loss = zero
        center_loss = zero
    total = cls_loss + 2.0 * reg_loss + 2.0 * iou_loss + center_loss
    return {
        "loss": total,
        "loss_cls": cls_loss,
        "loss_reg": reg_loss,
        "loss_iou": iou_loss,
        "loss_center": center_loss,
    }


def gaussian_soft_nms(predictions, sigma=0.5, score_threshold=0.001, max_detections=100):
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction["video_id"], prediction["label"])].append(dict(prediction))
    kept = []
    for group in grouped.values():
        group = sorted(group, key=lambda item: item["score"], reverse=True)[:500]
        selected = []
        while group and len(selected) < max_detections:
            group.sort(key=lambda item: item["score"], reverse=True)
            best = group.pop(0)
            selected.append(best)
            survivors = []
            for candidate in group:
                iou = segment_iou_np(best["segment"], candidate["segment"])
                candidate["score"] *= math.exp(-(iou * iou) / sigma)
                if candidate["score"] >= score_threshold:
                    survivors.append(candidate)
            group = survivors
        kept.extend(selected)
    return kept


@torch.no_grad()
def predict_dense(model, loader, device, score_threshold=0.01, topk=50):
    model.eval()
    predictions = []
    ground_truth = []
    for feats, mask, targets in loader:
        outputs = model(feats.to(device), mask.to(device))
        class_prob = outputs["pred_logits"].sigmoid()
        center_prob = outputs["pred_centerness"].sigmoid()[:, None]
        scores = class_prob * center_prob.sqrt()
        offsets = outputs["pred_offsets"]
        valid = ~outputs["mask"]
        length = scores.shape[-1]
        locations = (torch.arange(length, device=device).float() + 0.5) / length
        for batch_idx, target in enumerate(targets):
            flat_scores = scores[batch_idx].masked_fill(~valid[batch_idx][None], 0).flatten()
            keep_count = min(topk, int((flat_scores >= score_threshold).sum()))
            if keep_count:
                kept_scores, kept_indices = flat_scores.topk(keep_count)
                labels = torch.div(kept_indices, length, rounding_mode="floor")
                time_indices = kept_indices % length
                duration = float(target["duration"])
                window_start = float(target["window_start"])
                owner_start = float(target["owner_start"])
                owner_end = float(target["owner_end"])
                for score, label, time_idx in zip(kept_scores, labels, time_indices):
                    center = float(locations[time_idx])
                    global_center = window_start + center * duration
                    if not (owner_start <= global_center < owner_end):
                        continue
                    left = float(offsets[batch_idx, 0, time_idx])
                    right = float(offsets[batch_idx, 1, time_idx])
                    predictions.append(
                        {
                            "video_id": target["video_id"],
                            "segment": [
                                window_start + max(0.0, center - left) * duration,
                                window_start + min(1.0, center + right) * duration,
                            ],
                            "label": int(label),
                            "score": float(score),
                        }
                    )
            for segment, label in zip(target["global_segments"], target["labels"]):
                ground_truth.append(
                    {
                        "video_id": target["video_id"],
                        "segment": segment.tolist(),
                        "label": int(label),
                    }
                )
    return gaussian_soft_nms(predictions), ground_truth


def train_pseudo_detector(
    args,
    train_records,
    selection_records,
    test_records,
    labels,
    modality,
    output_dir,
):
    if modality == "wifi":
        window, train_stride, eval_stride, downsample = (
            args.wifi_window,
            args.wifi_stride,
            args.wifi_eval_stride,
            16,
        )
    else:
        window, train_stride, eval_stride, downsample = (
            args.wear_window,
            args.wear_stride,
            args.wear_eval_stride,
            8,
        )
    train_dataset = TADDataset(
        train_records,
        labels,
        modality,
        window_length=window,
        window_stride=train_stride,
        training=True,
    )
    selection_dataset = TADDataset(
        selection_records,
        labels,
        modality,
        window_length=window,
        window_stride=eval_stride,
        training=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.detector_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tad,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tad,
    )
    model = DenseTemporalDetector(
        input_dim=30 if modality == "wifi" else 600,
        num_classes=len(labels),
        modality=modality,
        hidden_dim=args.detector_hidden_dim,
        downsample=downsample,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.detector_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.detector_epochs, eta_min=1e-6)
    best = -1
    print(
        f"pseudo-detector=dense_anchor_free train_windows={len(train_dataset)} "
        f"selection_windows={len(selection_dataset)} postprocess=gaussian_soft_nms",
        flush=True,
    )
    for epoch in range(1, args.detector_epochs + 1):
        model.train()
        running = defaultdict(float)
        for feats, mask, targets in train_loader:
            feats, mask = feats.to(args.device), mask.to(args.device)
            targets = [
                {key: (value.to(args.device) if torch.is_tensor(value) else value) for key, value in target.items()}
                for target in targets
            ]
            losses = dense_detector_loss(model(feats, mask), targets, len(labels))
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in losses.items():
                running[key] += float(value.detach())
        scheduler.step()
        predictions, ground_truth = predict_dense(model, selection_loader, args.device)
        metrics = evaluate_map(predictions, ground_truth, len(labels))
        print(
            f"detector_epoch={epoch:03d} loss={running['loss'] / max(len(train_loader), 1):.4f} "
            f"avg_mAP={metrics['avg'] * 100:.2f} "
            + " ".join(f"mAP@{threshold}={metrics[threshold] * 100:.2f}" for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)),
            flush=True,
        )
        if metrics["avg"] > best:
            best = metrics["avg"]
            torch.save(
                {"model": model.state_dict(), "metrics": metrics, "epoch": epoch},
                output_dir / "detector_best.pt",
            )
    checkpoint = torch.load(output_dir / "detector_best.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    test_dataset = TADDataset(
        test_records,
        labels,
        modality,
        window_length=window,
        window_stride=eval_stride,
        training=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tad,
    )
    test_predictions, test_ground_truth = predict_dense(model, test_loader, args.device)
    test_metrics = evaluate_map(test_predictions, test_ground_truth, len(labels))
    print(
        f"FINAL_TEST_RESULT selected_epoch={checkpoint['epoch']} "
        f"selection_avg_mAP={checkpoint['metrics']['avg'] * 100:.2f} "
        f"test_avg_mAP={test_metrics['avg'] * 100:.2f} "
        + " ".join(
            f"test_mAP@{threshold}={test_metrics[threshold] * 100:.2f}"
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7)
        ),
        flush=True,
    )
    return {
        "selected_epoch": int(checkpoint["epoch"]),
        "selection_metrics": {
            str(key): float(value) for key, value in checkpoint["metrics"].items()
        },
        "test_metrics": {
            str(key): float(value) for key, value in test_metrics.items()
        },
    }


def split_strict_records(args, train_records, outer_test_records):
    if args.dataset == "wear":
        validation_subject = f"sbj_{(args.wear_split + 1) % 18}"
        selection_records = [
            record for record in train_records
            if record["video_id"] == validation_subject
        ]
        strict_train_records = [
            record for record in train_records
            if record["video_id"] != validation_subject
        ]
        if len(selection_records) != 1 or len(strict_train_records) != 16:
            raise RuntimeError(
                f"Unexpected Wear strict split: train={len(strict_train_records)} "
                f"selection={len(selection_records)}"
            )
    else:
        rng = np.random.RandomState(args.split_seed)
        indices = rng.permutation(len(train_records))
        selection_count = max(1, int(round(args.inner_val_fraction * len(indices))))
        selection_indices = set(indices[:selection_count].tolist())
        selection_records = [
            record for index, record in enumerate(train_records)
            if index in selection_indices
        ]
        strict_train_records = [
            record for index, record in enumerate(train_records)
            if index not in selection_indices
        ]
        validation_subject = None
    return strict_train_records, selection_records, outer_test_records, validation_subject


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dataset == "wifi":
        train_records, labels = load_wifi_records("train")
        outer_test_records, _ = load_wifi_records("test")
        modality = "wifi"
        run_name = "wifi"
    else:
        train_records, labels = load_wear_records(args.wear_split, "Training")
        outer_test_records, _ = load_wear_records(args.wear_split, "Validation")
        modality = "wear"
        run_name = f"wear_loso{args.wear_split}"
    train_records, selection_records, test_records, validation_subject = split_strict_records(
        args,
        train_records,
        outer_test_records,
    )
    if args.max_train_items:
        train_records = train_records[:args.max_train_items]
    if args.max_selection_items:
        selection_records = selection_records[:args.max_selection_items]
    if args.max_test_items:
        test_records = test_records[:args.max_test_items]
    budget_percent = int(round(args.boundary_fraction * 100))
    run_name = f"{run_name}_strict_boundary_{budget_percent:03d}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"dataset={run_name} train_records={len(train_records)} "
        f"selection_records={len(selection_records)} test_records={len(test_records)} "
        f"supervision=point(class+timestamp)+{budget_percent}%_boundaries "
        f"test_used_for_selection=0",
        flush=True,
    )
    print(
        f"protocol=inner_validation selection_metric=mAP "
        f"inner_validation_subject={validation_subject} "
        f"outer_test_evaluations=1 split_seed={args.split_seed}",
        flush=True,
    )
    localizer = train_localizer(args, train_records, labels, modality, output_dir)
    pseudo_records, quality, weak_quality, boundary_count = generate_pseudo_records(
        localizer,
        train_records,
        modality,
        args.device,
        boundary_fraction=args.boundary_fraction,
        seed=args.seed,
    )
    quality = np.asarray(quality)
    weak_quality = np.asarray(weak_quality)
    pseudo_metrics = {
        "mean_iou": float(quality.mean()),
        "median_iou": float(np.median(quality)),
        "recall_0.3": float((quality >= 0.3).mean()),
        "recall_0.5": float((quality >= 0.5).mean()),
        "recall_0.7": float((quality >= 0.7).mean()),
        "full_boundaries_used": int(boundary_count),
        "total_boundaries": int(len(quality)),
    }
    print(
        f"pseudo_quality boundary_fraction={args.boundary_fraction:.2f} full_boundaries_used={boundary_count} "
        f"mean_iou={quality.mean() * 100:.2f} median_iou={np.median(quality) * 100:.2f} "
        f"R@0.3={(quality >= 0.3).mean() * 100:.2f} R@0.5={(quality >= 0.5).mean() * 100:.2f} "
        f"R@0.7={(quality >= 0.7).mean() * 100:.2f}",
        flush=True,
    )
    if len(weak_quality):
        print(
            f"pseudo_quality_weak_only mean_iou={weak_quality.mean() * 100:.2f} "
            f"median_iou={np.median(weak_quality) * 100:.2f} "
            f"R@0.3={(weak_quality >= 0.3).mean() * 100:.2f} "
            f"R@0.5={(weak_quality >= 0.5).mean() * 100:.2f} "
            f"R@0.7={(weak_quality >= 0.7).mean() * 100:.2f}",
            flush=True,
        )
    detector_result = train_pseudo_detector(
        args,
        pseudo_records,
        selection_records,
        test_records,
        labels,
        modality,
        output_dir,
    )
    result = {
        "dataset": args.dataset,
        "wear_split": args.wear_split if args.dataset == "wear" else None,
        "boundary_fraction": float(args.boundary_fraction),
        "budget_percent": budget_percent,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_records": len(train_records),
        "selection_records": len(selection_records),
        "test_records": len(test_records),
        "inner_validation_subject": validation_subject,
        "pseudo_quality": pseudo_metrics,
        **detector_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("CURVE_RESULT", json.dumps(result, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser("Point-supervised sensor TAD")
    parser.add_argument("--dataset", choices=["wifi", "wear"], required=True)
    parser.add_argument("--wear-split", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/weak_sensor_tad")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--localizer-epochs", type=int, default=20)
    parser.add_argument("--localizer-batch-size", type=int, default=4)
    parser.add_argument("--localizer-hidden-dim", type=int, default=128)
    parser.add_argument("--localizer-lr", type=float, default=2e-4)
    parser.add_argument("--wifi-localizer-downsample", type=int, default=8)
    parser.add_argument("--wear-localizer-downsample", type=int, default=4)
    parser.add_argument("--detector-epochs", type=int, default=40)
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--detector-hidden-dim", type=int, default=192)
    parser.add_argument("--detector-lr", type=float, default=2e-4)
    parser.add_argument("--wifi-window", type=int, default=4096)
    parser.add_argument("--wifi-stride", type=int, default=2048)
    parser.add_argument("--wifi-eval-stride", type=int, default=2048)
    parser.add_argument("--wear-window", type=int, default=1024)
    parser.add_argument("--wear-stride", type=int, default=256)
    parser.add_argument("--wear-eval-stride", type=int, default=512)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-selection-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--boundary-fraction", type=float, default=0.0)
    parser.add_argument("--localizer-checkpoint", default=None)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=3407)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

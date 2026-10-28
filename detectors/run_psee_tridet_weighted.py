#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # Compatibility for the upstream ActionFormer metric code.

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from camera_baseline.tridet.libs.core.config import _update_config
from camera_baseline.tridet.libs.datasets.datasets import make_data_loader, make_dataset, register_dataset
from camera_baseline.tridet.libs.modeling import meta_archs as tridet_meta_archs
from camera_baseline.tridet.libs.modeling import make_meta_arch
from camera_baseline.tridet.libs.modeling.losses import ctr_diou_loss_1d, ctr_giou_loss_1d, sigmoid_focal_loss
from camera_baseline.tridet.libs.utils import ANETdetection, ModelEma, make_optimizer, make_scheduler, save_checkpoint, train_one_epoch, valid_one_epoch
from camera_baseline.tridet.libs.utils.train_utils import fix_random_seed


def truncate_feats_with_weights(
    data_dict,
    max_seq_len,
    trunc_thresh,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False,
):
    feat_len = data_dict["feats"].shape[1]
    num_segs = data_dict["segments"].shape[0]

    if feat_len <= max_seq_len:
        if crop_ratio is None:
            return data_dict
        max_seq_len = random.randint(
            max(round(crop_ratio[0] * feat_len), 1),
            min(round(crop_ratio[1] * feat_len), feat_len),
        )
        if feat_len == max_seq_len:
            return data_dict

    data_dict = copy.deepcopy(data_dict)

    for _ in range(max_num_trials):
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0], data_dict["segments"][:, 0])
        right = torch.minimum(window[:, 1], data_dict["segments"][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(data_dict["segments"][:, 1] - data_dict["segments"][:, 0])
        inter_ratio = inter / area_segs
        seg_idx = inter_ratio >= trunc_thresh

        if no_trunc:
            seg_trunc_idx = torch.logical_and((inter_ratio > 0.0), (inter_ratio < 1.0))
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            if seg_idx.sum().item() > 0:
                break
        else:
            break

    data_dict["feats"] = data_dict["feats"][:, st:ed].clone()
    data_dict["segments"] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    data_dict["segments"] = data_dict["segments"] - st
    data_dict["labels"] = data_dict["labels"][seg_idx].clone()
    if data_dict.get("weights") is not None:
        data_dict["weights"] = data_dict["weights"][seg_idx].clone()
    return data_dict


def weighted_forward(self, video_list):
    batched_inputs, batched_masks = self.preprocessing(video_list)

    feats, masks = self.backbone(batched_inputs, batched_masks)
    fpn_feats, fpn_masks = self.neck(feats, masks)
    points = self.point_generator(fpn_feats)
    out_cls_logits = self.cls_head(fpn_feats, fpn_masks)

    if self.use_trident_head:
        out_lb_logits = self.start_head(fpn_feats, fpn_masks)
        out_rb_logits = self.end_head(fpn_feats, fpn_masks)
    else:
        out_lb_logits = None
        out_rb_logits = None

    out_offsets = self.reg_head(fpn_feats, fpn_masks)
    out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
    out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
    fpn_masks = [x.squeeze(1) for x in fpn_masks]

    if all(x["segments"] is None or x["labels"] is None for x in video_list):
        zero = batched_inputs.sum() * 0.0
        losses = {"cls_loss": zero, "reg_loss": zero, "final_loss": zero}
    else:
        gt_segments = []
        gt_labels = []
        gt_weights = []
        for item in video_list:
            if item["segments"] is None or item["labels"] is None:
                gt_segments.append(torch.zeros((0, 2), device=self.device))
                gt_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))
                gt_weights.append(torch.zeros((0,), device=self.device))
            else:
                gt_segments.append(item["segments"].to(self.device))
                gt_labels.append(item["labels"].to(self.device))
                weights = item.get("weights")
                if weights is None:
                    weights = torch.ones_like(item["labels"], dtype=torch.float32)
                gt_weights.append(weights.to(self.device).float())
        gt_cls_labels, gt_offsets, gt_point_weights = self.label_points(points, gt_segments, gt_labels, gt_weights)
        losses = self.losses(
            fpn_masks,
            out_cls_logits,
            out_offsets,
            gt_cls_labels,
            gt_offsets,
            gt_point_weights,
            out_lb_logits,
            out_rb_logits,
        )

    if self.training:
        return losses

    results = self.inference(
        video_list,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        out_lb_logits,
        out_rb_logits,
    )
    return losses, results


@torch.no_grad()
def weighted_label_points(self, points, gt_segments, gt_labels, gt_weights):
    concat_points = torch.cat(points, dim=0)
    gt_cls, gt_offset, gt_weight = [], [], []

    for gt_segment, gt_label, gt_weight_single in zip(gt_segments, gt_labels, gt_weights):
        cls_targets, reg_targets, weight_targets = self.label_points_single_video(
            concat_points, gt_segment, gt_label, gt_weight_single
        )
        gt_cls.append(cls_targets)
        gt_offset.append(reg_targets)
        gt_weight.append(weight_targets)

    return gt_cls, gt_offset, gt_weight


@torch.no_grad()
def weighted_label_points_single_video(self, concat_points, gt_segment, gt_label, gt_weight):
    num_pts = concat_points.shape[0]
    num_gts = gt_segment.shape[0]

    if num_gts == 0:
        cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
        reg_targets = gt_segment.new_zeros((num_pts, 2))
        weight_targets = gt_segment.new_ones((num_pts,))
        return cls_targets, reg_targets, weight_targets

    lens = gt_segment[:, 1] - gt_segment[:, 0]
    lens = lens[None, :].repeat(num_pts, 1)

    gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
    left = concat_points[:, 0, None] - gt_segs[:, :, 0]
    right = gt_segs[:, :, 1] - concat_points[:, 0, None]
    reg_targets = torch.stack((left, right), dim=-1)

    if self.train_center_sample == "radius":
        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        t_mins = center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
        t_maxs = center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
        cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
        cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
        center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
        inside_gt_seg_mask = center_seg.min(-1)[0] > 0
    else:
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

    max_regress_distance = reg_targets.max(-1)[0]
    inside_regress_range = torch.logical_and(
        (max_regress_distance >= concat_points[:, 1, None]),
        (max_regress_distance <= concat_points[:, 2, None]),
    )

    lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
    lens.masked_fill_(inside_regress_range == 0, float("inf"))
    min_len, min_len_inds = lens.min(dim=1)

    min_len_mask = torch.logical_and(
        (lens <= (min_len[:, None] + 1e-3)), (lens < float("inf"))
    ).to(reg_targets.dtype)

    gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(reg_targets.dtype)
    cls_targets = min_len_mask @ gt_label_one_hot
    cls_targets.clamp_(min=0.0, max=1.0)
    reg_targets = reg_targets[range(num_pts), min_len_inds]
    reg_targets /= concat_points[:, 3, None]

    weight_targets = gt_weight[min_len_inds].to(reg_targets.dtype)
    weight_targets[min_len == float("inf")] = 1.0
    weight_targets.clamp_(min=0.0, max=1.0)
    return cls_targets, reg_targets, weight_targets


def weighted_losses(
    self,
    fpn_masks,
    out_cls_logits,
    out_offsets,
    gt_cls_labels,
    gt_offsets,
    gt_weights,
    out_start,
    out_end,
):
    valid_mask = torch.cat(fpn_masks, dim=1)

    if self.use_trident_head:
        out_start_logits = []
        out_end_logits = []
        for i in range(len(out_start)):
            x = (F.pad(out_start[i], (self.num_bins, 0), mode="constant", value=0)).unsqueeze(-1)
            x_size = list(x.size())
            x_size[-1] = self.num_bins + 1
            x_size[-2] = x_size[-2] - self.num_bins
            x_stride = list(x.stride())
            x_stride[-2] = x_stride[-1]
            x = x.as_strided(size=x_size, stride=x_stride)
            out_start_logits.append(x.permute(0, 2, 1, 3))

            x = (F.pad(out_end[i], (0, self.num_bins), mode="constant", value=0)).unsqueeze(-1)
            x = x.as_strided(size=x_size, stride=x_stride)
            out_end_logits.append(x.permute(0, 2, 1, 3))
    else:
        out_start_logits = None
        out_end_logits = None

    gt_cls = torch.stack(gt_cls_labels)
    point_weights = torch.stack(gt_weights).to(gt_cls.dtype)
    pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

    decoded_offsets = self.decode_offset_loss(out_offsets, out_start_logits, out_end_logits)
    decoded_offsets = decoded_offsets[pos_mask]

    if self.use_trident_head:
        pred_offsets = decoded_offsets[gt_cls[pos_mask].bool()]
        vid = torch.where(gt_cls[pos_mask])[0]
        gt_offsets = torch.stack(gt_offsets)[pos_mask][vid]
        reg_weights = point_weights[pos_mask][vid]
    else:
        pred_offsets = decoded_offsets
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        reg_weights = point_weights[pos_mask]

    num_pos = pos_mask.sum().item()
    self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
        1 - self.loss_normalizer_momentum
    ) * max(num_pos, 1)

    gt_target = gt_cls[valid_mask]
    gt_target *= 1 - self.train_label_smoothing
    gt_target += self.train_label_smoothing / (self.num_classes + 1)

    cls_loss = sigmoid_focal_loss(
        torch.cat(out_cls_logits, dim=1)[valid_mask],
        gt_target,
        reduction="none",
    )

    if self.use_trident_head and pred_offsets.numel() > 0:
        iou_rate = ctr_giou_loss_1d(pred_offsets, gt_offsets, reduction="none")
        rated_mask = gt_target > self.train_label_smoothing / (self.num_classes + 1)
        cls_loss[rated_mask] *= (1 - iou_rate) ** self.iou_weight_power

    cls_weight = point_weights[valid_mask].unsqueeze(-1)
    is_pos_point = (gt_cls.sum(-1) > 0)[valid_mask].unsqueeze(-1)
    cls_weight = torch.where(is_pos_point, cls_weight, torch.ones_like(cls_weight))
    cls_loss = (cls_loss * cls_weight).sum()
    cls_loss /= self.loss_normalizer

    if num_pos == 0:
        reg_loss = 0 * pred_offsets.sum()
    else:
        reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction="none")
        reg_loss = (reg_loss * reg_weights).sum()
        reg_loss /= self.loss_normalizer

    if self.train_loss_weight > 0:
        loss_weight = self.train_loss_weight
    else:
        loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

    final_loss = cls_loss + reg_loss * loss_weight
    return {"cls_loss": cls_loss, "reg_loss": reg_loss, "final_loss": final_loss}


def patch_tridet_reliability_weighting() -> None:
    tridet_meta_archs.TriDet.forward = weighted_forward
    tridet_meta_archs.TriDet.label_points = weighted_label_points
    tridet_meta_archs.TriDet.label_points_single_video = weighted_label_points_single_video
    tridet_meta_archs.TriDet.losses = weighted_losses


@register_dataset("psee_sensor")
class PseeSensorDataset(Dataset):
    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        sens_folder,
        sampling_rate,
        json_anno,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        tiou_thresholds,
        input_dim,
        num_classes,
        file_prefix,
        file_ext,
        force_upsampling,
    ):
        assert os.path.exists(feat_folder), feat_folder
        assert os.path.exists(json_anno), json_anno
        self.feat_folder = feat_folder
        self.sens_folder = sens_folder
        self.sampling_rate = sampling_rate
        self.file_prefix = file_prefix or ""
        self.file_ext = file_ext
        self.json_anno = json_anno
        self.db_attributes = {
            "dataset_name": "psee_sensor",
            "tiou_thresholds": tiou_thresholds,
            "empty_label_ids": [],
        }
        self.force_upsampling = force_upsampling
        self.split = split
        self.is_training = is_training
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.data_list, self.label_dict = self._load_json_db(json_anno)
        if len(self.label_dict) != num_classes:
            raise ValueError(f"num_classes={num_classes} but label_dict has {len(self.label_dict)}")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_anno):
        with open(json_anno, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        label_dict = {str(k): int(v) for k, v in json_data["label_dict"].items()}
        data = []
        for key, value in json_data["database"].items():
            if value["subset"].lower() not in self.split:
                continue
            feat_file = os.path.join(self.feat_folder, self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                raise FileNotFoundError(feat_file)
            fps = self.default_fps if self.default_fps is not None else float(value["fps"])
            duration = float(value.get("duration", 1e8))
            annotations = value.get("annotations", [])
            if annotations:
                segments, labels, weights = [], [], []
                for ann in annotations:
                    segments.append(ann["segment"])
                    labels.append(int(ann.get("label_id", label_dict[str(ann["label"])])))
                    weights.append(float(ann.get("weight", ann.get("reliability", 1.0))))
                segments_np = np.asarray(segments, dtype=np.float32)
                labels_np = np.asarray(labels, dtype=np.int64)
                weights_np = np.asarray(weights, dtype=np.float32)
            else:
                segments_np = None
                labels_np = None
                weights_np = None
            data.append(
                {
                    "id": key,
                    "fps": fps,
                    "duration": duration,
                    "segments": segments_np,
                    "labels": labels_np,
                    "weights": weights_np,
                }
            )
        return tuple(data), label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        filename = os.path.join(self.feat_folder, self.file_prefix + video_item["id"] + self.file_ext)
        feats = np.nan_to_num(np.load(filename).astype(np.float32))
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)
        if feats.shape[1] != self.input_dim:
            raise ValueError(f"{filename} has dim {feats.shape[1]}, expected {self.input_dim}")
        if self.force_upsampling:
            feat_stride = float((feats.shape[0] - 1) * self.feat_stride + self.num_frames) / self.max_seq_len
            num_frames = feat_stride
        else:
            feat_stride, num_frames = self.feat_stride, self.num_frames
            feats = feats[:: self.downsample_rate, :]
            feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * num_frames / feat_stride
        feats_t = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        if (feats_t.shape[-1] != self.max_seq_len) and self.force_upsampling:
            feats_t = F.interpolate(feats_t.unsqueeze(0), size=self.max_seq_len, mode="linear", align_corners=False).squeeze(0)
        if video_item["segments"] is not None:
            segments = torch.from_numpy(video_item["segments"] * video_item["fps"] / feat_stride - feat_offset)
            labels = torch.from_numpy(video_item["labels"])
            weights = torch.from_numpy(video_item["weights"])
        else:
            segments, labels, weights = None, None, None
        data_dict = {
            "video_id": video_item["id"],
            "feats": feats_t,
            "segments": segments,
            "labels": labels,
            "weights": weights,
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": self.num_frames,
        }
        if self.is_training and (segments is not None):
            data_dict = truncate_feats_with_weights(data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio)
        return data_dict


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_results_df(results: dict[str, Any]) -> pd.DataFrame:
    try:
        return pd.DataFrame(
            {
                "video-id": results["video-id"],
                "t-start": results["t-start"].tolist(),
                "t-end": results["t-end"].tolist(),
                "label": results["label"].tolist(),
                "score": results["score"].tolist(),
            }
        )
    except Exception:
        return pd.DataFrame(columns=["video-id", "t-start", "t-end", "label", "score"])


def evaluate(loader, model, cfg, split_name: str):
    loss, segments = valid_one_epoch(loader, model)
    tiou = np.asarray(cfg["dataset"]["tiou_thresholds"], dtype=float)
    evaluator = ANETdetection(cfg["dataset"]["json_anno"], split_name, tiou_thresholds=tiou, num_workers=4)
    try:
        mAP, _ = evaluator.evaluate(segments)
    except Exception as exc:
        print(f"EVAL_FAILED split={split_name} error={exc}", flush=True)
        mAP = np.zeros(len(tiou), dtype=float)
    return float(loss), np.asarray(mAP, dtype=float), segments


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg["init_rand_seed"] = args.seed
    cfg.setdefault("protocol", {})
    cfg["protocol"]["reliability_weighted_detector"] = True
    cfg["protocol"]["runner"] = "run_psee_tridet_weighted.py"
    cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    rng = fix_random_seed(args.seed, include_cuda=torch.cuda.is_available())
    cfg = _update_config(cfg)
    cfg_for_dump = copy.deepcopy(cfg)
    cfg_for_dump["device"] = str(cfg_for_dump["device"])
    (out_dir / "cfg.resolved.yaml").write_text(yaml.safe_dump(cfg_for_dump, sort_keys=False), encoding="utf-8")

    train_dataset = make_dataset(cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"])
    val_dataset = make_dataset(cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"])
    test_dataset = make_dataset(cfg["dataset_name"], False, cfg.get("test_split", ["testing"]), **cfg["dataset"])
    cfg["train_cfg"]["head_empty_cls"] = train_dataset.get_attributes()["empty_label_ids"]
    train_loader = make_data_loader(
        train_dataset,
        True,
        rng,
        cfg["loader"]["train_batch_size"],
        cfg["loader"]["num_workers"],
    )
    val_loader = make_data_loader(
        val_dataset,
        False,
        None,
        cfg["loader"]["test_batch_size"],
        cfg["loader"]["num_workers"],
    )
    test_loader = make_data_loader(
        test_dataset,
        False,
        None,
        cfg["loader"]["test_batch_size"],
        cfg["loader"]["num_workers"],
    )

    patch_tridet_reliability_weighting()
    model = make_meta_arch(cfg["model"]["model_name"], **cfg["model"]).to(cfg["device"])
    print(
        f"START dataset={Path(args.config).parent.name} split={Path(args.config).stem} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"device={cfg['device']} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    optimizer = make_optimizer(model, cfg["opt"])
    scheduler = make_scheduler(optimizer, cfg["opt"], len(train_loader))
    model_ema = ModelEma(model)

    max_epochs = int(cfg["opt"].get("early_stop_epochs", cfg["opt"]["epochs"] + cfg["opt"].get("warmup_epochs", 0)))
    best = {
        "epoch": -1,
        "val_avg": -1.0,
        "val_mAP": None,
        "state_dict": None,
    }
    history = []
    for epoch in range(max_epochs):
        start = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, model_ema, cfg["train_cfg"]["clip_grad_l2norm"])
        val_loss, val_mAP, _ = evaluate(val_loader, model, cfg, "validation")
        val_avg = float(np.mean(val_mAP))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": val_loss,
                "val_mAP": val_mAP.tolist(),
                "val_avg": val_avg,
                "seconds": time.time() - start,
            }
        )
        print(
            f"EPOCH {epoch+1:03d}/{max_epochs:03d} train_loss={train_loss:.4f} "
            f"val_avg={val_avg*100:.2f} val_mAP={[round(x*100,2) for x in val_mAP]}",
            flush=True,
        )
        if val_avg > float(best["val_avg"]):
            best = {
                "epoch": epoch + 1,
                "val_avg": val_avg,
                "val_mAP": val_mAP.tolist(),
                "state_dict": cpu_state_dict(model),
            }
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": best["state_dict"],
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "state_dict_ema": cpu_state_dict(model_ema.module),
                    "val_mAP": best["val_mAP"],
                    "val_avg": best["val_avg"],
                },
                True,
                str(out_dir),
                "checkpoint_best.pth.tar",
            )

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    test_loss, test_mAP, test_segments = evaluate(test_loader, model, cfg, "testing")
    to_results_df(test_segments).to_csv(out_dir / "test_predictions.csv", index=False)
    result = {
        "config": args.config,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_validation_mAP": best["val_mAP"],
        "best_validation_avg": best["val_avg"],
        "test_loss": test_loss,
        "test_mAP": test_mAP.tolist(),
        "test_avg": float(np.mean(test_mAP)),
        "tiou_thresholds": cfg["dataset"]["tiou_thresholds"],
        "history": history,
        "protocol": cfg.get("protocol", {}),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "RESULT "
        f"best_epoch={best['epoch']} "
        f"test_avg={float(np.mean(test_mAP))*100:.2f} "
        f"test_mAP={[round(x*100,2) for x in test_mAP]} "
        f"out={out_dir / 'result.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()

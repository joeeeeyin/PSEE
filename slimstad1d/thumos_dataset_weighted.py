import csv
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from STAD.common import videotransforms
from STAD.common.config import config
from STAD.common.thumos_dataset import (
    get_class_index_map,
    load_video_data,
    split_videos,
)


def _load_reliability_rows(reliability_path):
    import json

    with open(reliability_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload.get("rows", payload)
    weights = []
    for row in rows:
        segment_reliability = float(row.get("reliability", 1.0))
        left = float(
            row.get(
                "left_confidence",
                row.get("left_reliability", row.get("r_left", segment_reliability)),
            )
        )
        right = float(
            row.get(
                "right_confidence",
                row.get("right_reliability", row.get("r_right", segment_reliability)),
            )
        )
        weights.append((left, right, segment_reliability))
    return weights


def get_video_anno_weighted(video_infos, video_anno_path, reliability_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    reliability_rows = _load_reliability_rows(reliability_path)
    if len(reliability_rows) != len(df_anno):
        raise RuntimeError(
            f"Reliability row count mismatch: {len(reliability_rows)} vs {len(df_anno)}"
        )

    originidx_to_idx, _ = get_class_index_map(config["dataset"].get("class_info_path"))
    video_annos = {}
    for anno, reliability in zip(df_anno, reliability_rows):
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]["count"]
        sample_count = video_infos[video_name]["sample_count"]
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        left_conf, right_conf, segment_conf = reliability
        item = [
            start_gt,
            end_gt,
            class_idx,
            float(left_conf),
            float(right_conf),
            float(segment_conf),
        ]
        video_annos.setdefault(video_name, []).append(item)
    return video_annos


def split_videos_weighted(video_infos, video_annos, clip_length, stride):
    training_list = []
    min_anno_dict = {}
    for video_name in video_annos.keys():
        min_anno = clip_length
        sample_count = video_infos[video_name]["sample_count"]
        annos = video_annos[video_name]
        if sample_count <= clip_length:
            offsetlist = [0]
            min_anno_len = min([x[1] - x[0] for x in annos])
            min_anno = min(min_anno, min_anno_len)
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]

        for offset in offsetlist:
            left, right = offset + 1, offset + clip_length
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa >= 0.5:
                    cur_annos.append([
                        max(anno[0] - offset, 1),
                        min(anno[1] - offset, clip_length),
                        anno[2],
                        anno[3],
                        anno[4],
                        anno[5],
                    ])
            if len(cur_annos) > 0:
                min_anno_len = min([x[1] - x[0] for x in cur_annos])
                min_anno = min(min_anno, min_anno_len)
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                start_weight = np.ones([clip_length], dtype=np.float32)
                end_weight = np.ones([clip_length], dtype=np.float32)
                for anno in cur_annos:
                    s, e, _class_id = anno[:3]
                    left_conf = float(anno[3]) if len(anno) > 3 else 1.0
                    right_conf = float(anno[4]) if len(anno) > 4 else left_conf
                    d = max((e - s) / 10.0, 2.0)
                    start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
                    prev_start = start[start_s:start_e] > 0
                    start[start_s:start_e] = 1
                    start_weight[start_s:start_e] = np.where(
                        prev_start,
                        np.maximum(start_weight[start_s:start_e], np.float32(left_conf)),
                        np.float32(left_conf),
                    )
                    end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
                    prev_end = end[end_s:end_e] > 0
                    end[end_s:end_e] = 1
                    end_weight[end_s:end_e] = np.where(
                        prev_end,
                        np.maximum(end_weight[end_s:end_e], np.float32(right_conf)),
                        np.float32(right_conf),
                    )
                training_list.append({
                    "video_name": video_name,
                    "offset": offset,
                    "annos": cur_annos,
                    "start": start,
                    "end": end,
                    "start_weight": start_weight,
                    "end_weight": end_weight,
                })
        min_anno_dict[video_name] = math.ceil(min_anno)
    return training_list, min_anno_dict


def annos_transform_weighted(annos, clip_length):
    targets = []
    weights = []
    for anno in annos:
        targets.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2],
        ])
        if len(anno) >= 5:
            weights.append([float(anno[3]), float(anno[4])])
        elif len(anno) >= 4:
            weights.append([float(anno[3]), float(anno[3])])
        else:
            weights.append([1.0, 1.0])
    return targets, weights


class THUMOSWeighted_Dataset(Dataset):
    def __init__(
        self,
        data_dict,
        video_infos,
        video_annos,
        clip_length=config["dataset"]["training"]["clip_length"],
        crop_size=config["dataset"]["training"]["crop_size"],
        stride=config["dataset"]["training"]["clip_stride"],
        rgb_norm=False,
        training=True,
        origin_ratio=0.5,
    ):
        self.training_list, self.th = split_videos_weighted(
            video_infos,
            video_annos,
            clip_length,
            stride,
        )
        self.data_dict = data_dict
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.random_crop = videotransforms.RandomCrop(crop_size)
        self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
        self.center_crop = videotransforms.CenterCrop(crop_size)
        self.rgb_norm = rgb_norm
        self.training = training
        self.origin_ratio = origin_ratio

    def __len__(self):
        return len(self.training_list)

    def get_bg(self, annos, min_action):
        annos = [[anno[0], anno[1]] for anno in annos]
        times = []
        for anno in annos:
            times.extend(anno)
        times.extend([0, self.clip_length - 1])
        times.sort()
        regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
        regions = list(filter(
            lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action,
            regions,
        ))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]

    def augment_(self, input_data, annos, th):
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
        except IndexError:
            return input_data, annos, False
        gt_len = gt[1] - gt[0]
        region = range(math.floor(th), math.ceil(gt_len - th))
        t = random.choice(region) + math.ceil(gt[0])
        l_len = math.ceil(t - gt[0])
        r_len = math.ceil(gt[1] - t)
        try:
            bg = self.get_bg(annos, th)
        except IndexError:
            return input_data, annos, False
        start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
        end_idx = start_idx + th

        new_input = input_data.clone()
        if gt[1] < start_idx:
            new_input[:, t:t + th, ] = input_data[:, start_idx:end_idx, ]
            new_input[:, t + th:end_idx, ] = input_data[:, t:start_idx, ]
            new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
        else:
            new_input[:, start_idx:t - th] = input_data[:, end_idx:t, ]
            new_input[:, t - th:t, ] = input_data[:, start_idx:end_idx, ]
            new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
        return new_input, new_annos, True

    def augment(self, input_data, annos, th, max_iter=10):
        flag = True
        i = 0
        while flag and i < max_iter:
            new_input, new_annos, flag = self.augment_(input_data, annos, th)
            i += 1
        return new_input, new_annos, flag

    def __getitem__(self, idx):
        sample_info = self.training_list[idx]
        video_data = self.data_dict[sample_info["video_name"]]
        offset = sample_info["offset"]
        annos_weighted = sample_info["annos"]
        annos_plain = [[anno[0], anno[1], anno[2]] for anno in annos_weighted]
        th = self.th[sample_info["video_name"]]

        input_data = video_data[:, offset: offset + self.clip_length]
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)

        input_data = torch.from_numpy(input_data).float()
        if self.rgb_norm:
            input_data = (input_data / 255.0) * 2.0 - 1.0
        ssl_input_data, ssl_annos, flag = self.augment(input_data, annos_plain, th, 1)
        target, weights = annos_transform_weighted(annos_weighted, self.clip_length)
        target = np.stack(target, 0)
        ssl_target = np.stack(ssl_annos, 0)
        target_weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))

        scores = np.stack(
            [
                sample_info["start"],
                sample_info["end"],
                sample_info.get("start_weight", np.ones([self.clip_length], dtype=np.float32)),
                sample_info.get("end_weight", np.ones([self.clip_length], dtype=np.float32)),
            ],
            axis=0,
        )
        scores = torch.from_numpy(scores.copy()).float()
        return input_data, target, scores, ssl_input_data, ssl_target, flag, target_weights


def detection_collate_weighted(batch):
    targets = []
    clips = []
    scores = []
    ssl_targets = []
    ssl_clips = []
    flags = []
    target_weights = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        scores.append(sample[2])
        ssl_clips.append(sample[3])
        ssl_targets.append(torch.FloatTensor(sample[4]))
        flags.append(sample[5])
        target_weights.append(sample[6].float())
    return (
        torch.stack(clips, 0),
        targets,
        torch.stack(scores, 0),
        torch.stack(ssl_clips, 0),
        ssl_targets,
        flags,
        target_weights,
    )

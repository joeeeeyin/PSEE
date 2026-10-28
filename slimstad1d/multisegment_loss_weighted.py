import torch
import torch.nn as nn
import torch.nn.functional as F

from STAD.common.config import config
from STAD.thumos14.multisegment_loss import iou_loss


def focal_loss_weighted(prob, target, num_classes, reliability=None, gamma=2.0):
    target = target.view(-1).long()
    pt = prob.gather(1, target[:, None]).clamp(min=1e-6).view(-1)
    alpha = torch.full((num_classes,), 0.75, device=prob.device)
    alpha[0] = 0.25
    alpha_t = alpha.gather(0, target)
    loss = -alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)
    if reliability is not None:
        loss = loss * reliability.view(-1)
    return loss.sum()


class WeightedMultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, negpos_ratio, use_gpu=True,
                 use_focal_loss=False):
        super().__init__()
        self.num_classes = num_classes
        print("num_classes:", self.num_classes)
        self.overlap_thresh = overlap_thresh
        self.negpos_ratio = negpos_ratio
        self.use_gpu = use_gpu
        self.use_focal_loss = use_focal_loss
        self.center_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, predictions, targets, pre_locs=None, target_weights=None):
        loc_data, conf_data, prop_loc_data, prop_conf_data, center_data, priors = predictions
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        clip_length = config["dataset"]["training"]["clip_length"]

        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
        prop_loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        prop_conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
        boundary_reliability_t = torch.ones(num_batch, num_priors, 2, device=loc_data.device)

        with torch.no_grad():
            for idx in range(num_batch):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                if target_weights is None:
                    gt_boundary_reliability = torch.ones(labels.size(0), 2, dtype=torch.float32, device=loc_data.device)
                else:
                    gt_boundary_reliability = target_weights[idx].to(loc_data.device).float()
                    if gt_boundary_reliability.dim() == 1:
                        gt_boundary_reliability = gt_boundary_reliability[:, None].expand(-1, 2)
                    elif gt_boundary_reliability.size(-1) == 1:
                        gt_boundary_reliability = gt_boundary_reliability.expand(-1, 2)
                    else:
                        gt_boundary_reliability = gt_boundary_reliability[:, :2]
                    gt_boundary_reliability = gt_boundary_reliability.clamp(0.05, 1.0)

                pre_loc = loc_data[idx]
                k = priors.size(0)
                n = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(k, n)
                left = (center - truths[:, 0].unsqueeze(0).expand(k, n)) * clip_length
                right = (truths[:, 1].unsqueeze(0).expand(k, n) - center) * clip_length
                area = left + right
                maxn = clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn
                best_truth_area, best_truth_idx = area.min(1)

                loc_t[idx][:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
                loc_t[idx][:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
                conf = labels[best_truth_idx]
                conf[best_truth_area >= maxn] = 0
                conf_t[idx] = conf
                boundary_reliability_t[idx] = gt_boundary_reliability[best_truth_idx]
                boundary_reliability_t[idx][best_truth_area >= maxn] = 1.0

                iou = iou_loss(pre_loc, loc_t[idx], loss_type="calc iou")
                prop_conf = conf.clone()
                prop_conf[iou < self.overlap_thresh] = 0
                prop_conf_t[idx] = prop_conf
                prop_w = pre_loc[:, 0] + pre_loc[:, 1]
                prop_loc_t[idx][:, 0] = (loc_t[idx][:, 0] - pre_loc[:, 0]) / (0.5 * prop_w)
                prop_loc_t[idx][:, 1] = (loc_t[idx][:, 1] - pre_loc[:, 1]) / (0.5 * prop_w)

        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        gt_loc_t = loc_t.clone()
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_target = loc_t[pos_idx].view(-1, 2)
        pos_boundary_w = boundary_reliability_t[pos].view(-1, 2)
        pos_w = pos_boundary_w.mean(dim=1)

        if loc_p.numel() > 0:
            loss_l_vec = iou_loss(loc_p, loc_target, loss_type="giou", reduction="none")
            loss_l = (loss_l_vec * pos_w).sum()
        else:
            loss_l = loc_p.sum()

        prop_pos = prop_conf_t > 0
        prop_pos_idx = prop_pos.unsqueeze(-1).expand_as(prop_loc_data)
        prop_loc_p = prop_loc_data[prop_pos_idx].view(-1, 2)
        prop_loc_target = prop_loc_t[prop_pos_idx].view(-1, 2)
        prop_boundary_w_pos = boundary_reliability_t[prop_pos].view(-1, 2)
        prop_w_pos = prop_boundary_w_pos.mean(dim=1)

        if prop_loc_p.numel() > 0:
            loss_prop_l_vec = F.l1_loss(prop_loc_p, prop_loc_target, reduction="none")
            loss_prop_l = (loss_prop_l_vec * prop_boundary_w_pos).sum()
        else:
            loss_prop_l = prop_loc_p.sum()

        prop_pre_loc = loc_data[pos_idx].view(-1, 2)
        cur_loc_t = gt_loc_t[pos_idx].view(-1, 2)
        prop_loc_p_refine = prop_loc_data[pos_idx].view(-1, 2)
        center_p = center_data[pos.unsqueeze(pos.dim())].view(-1)
        if prop_pre_loc.numel() > 0:
            prop_pre_w = (prop_pre_loc[:, 0] + prop_pre_loc[:, 1]).unsqueeze(-1)
            cur_loc_p = 0.5 * prop_pre_w * prop_loc_p_refine + prop_pre_loc
            ious = iou_loss(cur_loc_p, cur_loc_t, loss_type="calc iou").clamp_(min=0)
            center_vec = F.binary_cross_entropy_with_logits(center_p, ious, reduction="none")
            loss_ct = (center_vec * pos_w).sum()
        else:
            loss_ct = prop_pre_loc.sum()

        conf_prob = F.softmax(conf_data.view(-1, num_classes), dim=1)
        loss_c = focal_loss_weighted(
            conf_prob,
            conf_t.view(-1, 1),
            num_classes,
            reliability=None,
        )

        prop_conf_prob = F.softmax(prop_conf_data.view(-1, num_classes), dim=1)
        loss_prop_c = focal_loss_weighted(
            prop_conf_prob,
            prop_conf_t.view(-1, 1),
            num_classes,
            reliability=None,
        )

        n = torch.clamp(pos_w.sum(), min=1.0)
        pn = torch.clamp(prop_w_pos.sum(), min=1.0)
        cls_n = torch.clamp(pos.sum().float(), min=1.0)
        prop_cls_n = torch.clamp(prop_pos.sum().float(), min=1.0)
        loss_l /= n
        loss_c /= cls_n
        loss_prop_l /= pn
        loss_prop_c /= prop_cls_n
        loss_ct /= n
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct

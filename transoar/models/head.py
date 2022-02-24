import math

import torch
import torch.nn as nn

from transoar.models.loss import BCEWithLogitsLossOneHot, GIoULoss, SoftDiceLoss

    
class ClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._num_classes = config['classifier_classes']
        out_channels = config['anchors_per_position'] * config['classifier_classes']

        block_1 = [
            nn.Conv3d(config['cls_reg_channels'], config['head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, config['head_channels'], eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        ]

        block_2 = [
            nn.Conv3d(config['head_channels'], config['head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, config['head_channels'], eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        ]

        block_3 = [
            nn.Conv3d(config['head_channels'], out_channels, kernel_size=3, stride=1, padding=1),
        ]

        # Classification head
        self._body = nn.Sequential(*block_1, *block_2)
        self._out = nn.Sequential(*block_3)
        self._init_weights()

        # Loss fct
        self._loss = BCEWithLogitsLossOneHot(
            num_classes=config['classifier_classes'], weight=None, loss_weight=config['loss_weight']
        )

        self._logits_to_preds = nn.Sigmoid()

    def _init_weights(self):
        prior_prob = self._config['prior_prob']
        if prior_prob is not None:
            for layer in self.modules():
                if isinstance(layer, nn.Conv3d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for layer in self._out.modules():
                if isinstance(layer, nn.Conv3d):
                    torch.nn.init.constant_(layer.bias, bias_value)

    def forward(self, x):
        cls_logits = self._out(self._body(x))   # [N, C, Y, X, Z]

        cls_logits = cls_logits.permute(0, 2, 3, 4, 1)
        cls_logits = cls_logits.contiguous()
        cls_logits = cls_logits.view(x.size()[0], -1, self._num_classes) # [N, anchors, num_classes]

        return cls_logits

    def compute_loss(self, pred_logits, targets):
        return self._loss(pred_logits, targets.long())

    def box_logits_to_probs(self, box_logits):
        return self._logits_to_preds(box_logits)
    
class RegHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._learnable_scale = config['learnabel_scale']
        out_channels = config['anchors_per_position'] * 6

        block_1 = [
            nn.Conv3d(config['cls_reg_channels'], config['head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, config['head_channels'], eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        ]

        block_2 = [
            nn.Conv3d(config['head_channels'], config['head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, config['head_channels'], eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        ]

        block_3 = [
            nn.Conv3d(config['head_channels'], out_channels, kernel_size=3, stride=1, padding=1),
        ]

        # Regression head
        self._body = nn.Sequential(*block_1, *block_2)
        self._out = nn.Sequential(*block_3)
        self._init_weights()

        # Learnable scale
        if self._learnable_scale:
            num_levels = len(config['input_levels'])
            self._scales = nn.ModuleList([Scale() for _ in range(num_levels)])

        # Loss fct
        self._loss = GIoULoss(loss_weight=config['loss_weight'])

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x, level):
        reg_logits = self._out(self._body(x))   # [N, C, Y, X, Z]

        if self._learnable_scale:
            reg_logits = self._scales[level](reg_logits)

        reg_logits = reg_logits.permute((0, 2, 3, 4, 1))
        reg_logits = reg_logits.contiguous()
        reg_logits = reg_logits.view(x.size()[0], -1, 6)    # [N, anchors, 6] 

        return reg_logits

    def compute_loss(self, pred_deltas, target_deltas):
        return self._loss(pred_deltas, target_deltas)

class Scale(nn.Module):
    def __init__(self, scale: float = 1.):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, inp):
        return inp * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale.item()}"

class SegHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._in_channels = config['seg_channels']

        self._out = nn.Conv3d(self._in_channels, 2, kernel_size=1, stride=1)

        self._dice_loss = SoftDiceLoss(
            nonlin=torch.nn.Softmax(dim=1), batch_dice=True, smooth_nom=1e-05, smooth_denom=1e-05,
            do_bg=False
        )

        self._ce_loss = nn.CrossEntropyLoss()

        self._logits_convert_fn = nn.Softmax(dim=1)
        self._alpha = 0.5

    def forward(self, x):
        x = x['P0']
        return {"seg_logits": self._out(x)}

    def compute_loss(self, pred_seg, target):
        # To only predict if voxel fg or bg
        target[target > 0] = 1
        seg_logits = pred_seg["seg_logits"]
        return {
            "seg_ce": self._alpha * self._ce_loss(seg_logits, target.long()),
            "seg_dice": (1 - self._alpha) * self._dice_loss(seg_logits, target),
            }

    def postprocess_for_inference(self, prediction):
        return {"pred_seg": self._logits_convert_fn(prediction["seg_logits"])}


class DetectionHeadHNMNative(nn.Module):
    def __init__(self, classifier, regressor, coder, sampler):
        super().__init__()

        self.classifier = classifier
        self.regressor = regressor
        self.coder = coder
        self.fg_bg_sampler = sampler

    def forward(self, fmaps):
        logits, offsets = [], []
        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p))
            offsets.append(self.regressor(p, level=level))

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)
        return {"box_deltas": box_deltas, "box_logits": box_logits}


    def postprocess_for_inference(self, prediction, anchors):
        postprocess_predictions = {
            "pred_boxes": self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs": self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions
    

    def compute_loss(self, prediction, target_labels, matched_gt_boxes, anchors):
        box_logits, box_deltas = prediction["box_logits"], prediction["box_deltas"]

        with torch.no_grad():
            losses = {}
            sampled_pos_inds, sampled_neg_inds = self.select_indices(target_labels, box_logits)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
                ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(self, target_labels, boxes_scores):
        boxes_max_fg_probs = self.classifier.box_logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(dim=1)[0]  # search max of fg probs

        # positive and negative anchor indices per image
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        return sampled_pos_inds, sampled_neg_inds

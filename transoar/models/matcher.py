"""Module containing the matcher."""

import torch
from torch import nn

from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz, generalized_bbox_iou_3d


class Matcher(nn.Module):
    def __init__(
        self, cost_class=1, cost_bbox=1, cost_giou=1, anchor_matching=True, num_organs=None
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.anchor_matching = anchor_matching
        self.num_organs = num_organs

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, anchors, num_top_queries=1):
        bs, num_queries, _ = outputs["pred_logits"].shape
        num_queries_per_organ = int(num_queries / self.num_organs)

        # Split queries in individual classes
        if self.anchor_matching:
            classes_queries_boxes = anchors[None].repeat((bs, 1, 1)).reshape(bs, self.num_organs, num_queries_per_organ, -1).cpu().float() 
        else:
            classes_queries_boxes = outputs["pred_boxes"].reshape(bs, self.num_organs, num_queries_per_organ, -1).cpu().float()
        classes_queries_probs = outputs["pred_logits"].reshape(bs, self.num_organs, num_queries_per_organ, -1).cpu().float()

        # Get targets
        tgt = [{label.item(): box.cpu() for box, label in zip(target['boxes'], target['labels'])} for target in targets]

        # Generate soft query labels based on IoU with target
        soft_labels = torch.zeros_like(classes_queries_probs).squeeze(-1)
        matches = torch.zeros_like(classes_queries_probs, dtype=torch.long).squeeze(-1)

        for batch, (batch_pred_logits, batch_pred_boxes) in enumerate(zip(classes_queries_probs, classes_queries_boxes)):
            for class_, (class_pred_logits, class_pred_boxes) in enumerate(zip(batch_pred_logits, batch_pred_boxes), 1):
                try:
                    tgt_box = tgt[batch][class_]
                except KeyError:
                    soft_labels[batch, class_ -1] = -1
                    continue

                # Determine cost based on different metrices
                cost_class = -class_pred_logits.sigmoid().squeeze()
                cost_bbox = torch.cdist(class_pred_boxes, tgt_box[None], p=1).squeeze()
                cost_giou = -generalized_bbox_iou_3d(box_cxcyczwhd_to_xyzxyz(class_pred_boxes.clip(min=0)), box_cxcyczwhd_to_xyzxyz(tgt_box[None])).squeeze()

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                best_query_ids = torch.topk(C, num_top_queries, largest=False)[-1]

                try:
                    for query_id in best_query_ids:
                        matches[batch, class_ -1, query_id] = 1
                    soft_labels[batch,  class_ - 1] = ((cost_giou - cost_giou.max()) / (cost_giou.min() - cost_giou.max())).clip(min=0) # nomalize
                except TypeError:
                    matches[batch, class_ -1, 0] = 1
                    soft_labels[batch,  class_ - 1] = 1

        return matches, soft_labels

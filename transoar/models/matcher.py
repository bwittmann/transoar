"""Module containing the hungarian matcher, adapted from https://github.com/facebookresearch/detr."""

import torch
from torch import nn

from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz, generalized_bbox_iou_3d


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_center_dist: float = 1,
        anchor_matching: bool = True
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_center_dist = cost_center_dist
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_center_dist != 0, "all costs can't be 0"

        self.anchor_matching = anchor_matching

    @torch.no_grad()
    def forward(self, outputs, targets, anchors, num_top_queries=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs = outputs["pred_logits"].shape[0]

        # Generate soft query labels based on IoU with target
        soft_labels = torch.zeros_like(outputs['pred_logits'][..., 0])

        # Split queries in individual classes
        if self.anchor_matching:
            classes_queries_boxes = anchors[None].repeat((bs, 1, 1)).reshape(bs, 20, 27, -1) #TODO: don't hardcode any information
        else:
            classes_queries_boxes = outputs["pred_boxes"].reshape(bs, 20, 27, -1)
        classes_queries_probs = outputs["pred_logits"].reshape(bs, 20, 27, -1)

        # Get targets
        tgt = [{label.item(): box for box, label in zip(target['boxes'], target['labels'])} for target in targets]

        matches = torch.zeros((bs, 20, 27), dtype=torch.long)
        for batch, (batch_pred_logits, batch_pred_boxes) in enumerate(zip(classes_queries_probs, classes_queries_boxes)):
            for class_, (class_pred_logits, class_pred_boxes) in enumerate(zip(batch_pred_logits, batch_pred_boxes), 1):
                try:
                    tgt_box = tgt[batch][class_]
                except KeyError:
                    soft_labels[batch, class_ -1: class_ -1 + 27] = -1  # class not in tgt
                    continue

                # Determine cost based on different metrices
                cost_class = -class_pred_logits.sigmoid().squeeze()
                cost_bbox = torch.cdist(class_pred_boxes, tgt_box[None], p=1).squeeze()
                cost_center_dist = torch.cdist(class_pred_boxes[:, :3], tgt_box[:3][None], p=2).squeeze()
                cost_giou = -generalized_bbox_iou_3d(box_cxcyczwhd_to_xyzxyz(class_pred_boxes.clip(min=0)), box_cxcyczwhd_to_xyzxyz(tgt_box[None])).squeeze()

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_center_dist * cost_center_dist
                best_query_ids = torch.topk(C, num_top_queries, largest=False)[-1]

                # Assign soft labels and match
                soft_labels[batch,  (class_ - 1) * 27 : class_ * 27] = ((cost_giou - cost_giou.max()) / (cost_giou.min() - cost_giou.max())).clip(min=0) # nomalize

                for query_id in best_query_ids:
                    matches[batch, class_ -1, query_id] = 1

        return matches, soft_labels
    
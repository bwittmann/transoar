"""Module containing the hungarian matcher, adapted from https://github.com/facebookresearch/detr."""

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

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
        anchor_matching: bool = False
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
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

        self.anchor_matching = anchor_matching

    @torch.no_grad()
    def forward(self, outputs, targets, anchors):
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

        # Split queries in individual classes   TODO: don't hardcode any information
        if self.anchor_matching:
            classes_queries_boxes = [torch.split(anchors, 27 * 3, dim=0) for _ in range(bs)]
        else:
            classes_queries_boxes = [torch.split(batch_boxes, 27 * 3, dim=0) for batch_boxes in torch.unbind(outputs["pred_boxes"], dim=0)]

        classes_queries_probs = [[logits.softmax(-1) for logits in torch.split(batch_logits, 27 * 3, dim=0)] for batch_logits in torch.unbind(outputs["pred_logits"], dim=0)]
        assert len(classes_queries_probs[0]) == 20 and len(classes_queries_boxes[0]) == 20

        # Get targets
        tgt_ids = [v["labels"] for v in targets]
        tgt_boxes = [v["boxes"] for v in targets]

        matches = []
        for batch in range(bs):
            batch_matches = []
            batch_tgt_ids = tgt_ids[batch]
            batch_tgt_boxes = tgt_boxes[batch]

            batch_queries_probs = classes_queries_probs[batch]
            batch_queries_boxes = classes_queries_boxes[batch]

            for tgt_id, tgt_box in zip(batch_tgt_ids, batch_tgt_boxes):
                tgt_id = tgt_id - 1  # Since class 0 has id 1
                class_queries_boxes = batch_queries_boxes[tgt_id]
                class_queries_probs = batch_queries_probs[tgt_id]

                # Determine individual costs
                cost_class = -class_queries_probs[:, -1]
                cost_bbox = torch.cdist(class_queries_boxes, tgt_box[None], p=1).squeeze()
                cost_giou = -generalized_bbox_iou_3d(box_cxcyczwhd_to_xyzxyz(class_queries_boxes.clip(min=0)), box_cxcyczwhd_to_xyzxyz(tgt_box[None])).squeeze()

                # Determine final cost and best performing query
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                best_query = C.argmin() # TODO: add hard negative or dropout
                batch_matches.append([tgt_id.item(), (best_query + tgt_id * 27 * 3).item()])

            matches.append(batch_matches)

        return matches
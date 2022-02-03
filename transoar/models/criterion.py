"""Module containing the loss functions of the transoar project."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.utils.bboxes import generalized_bbox_iou_3d, box_cxcyczwhd_to_xyzxyz

class TransoarCriterion(nn.Module):
    """ This class computes the loss for TransoarNet.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        
        # Hack to make deterministic, https://github.com/pytorch/pytorch/issues/46024
        self.cls_weights = torch.tensor(
            [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        ).type(torch.FloatTensor)

    def loss_class(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        src_idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"] for t in targets])
        src_logits_matched = src_logits[src_idx]

        loss_ce = F.cross_entropy(src_logits_matched, target_classes_o) # TODO: operates on logits?

        # Peak loss
        loss_peak = torch.tensor(0.).to(device='cuda')
        # for batch in range(src_logits.shape[0]):
        #     batch_logits = src_logits[batch]
        #     classes_logits = [logits.softmax(-1) for logits in torch.split(batch_logits, 27 * 3, dim=0)]    # TODO: dont hardcode.
        #     matched_query_ids = indices[batch]

        #     for matched_query_id in matched_query_ids:
        #         tgt_class, query_id = matched_query_id
        #         class_logits = classes_logits[tgt_class]

        #         matched_query_id_offset = query_id - tgt_class * 27 * 3
        #         loss_peak += F.cross_entropy(class_logits[:, tgt_class][None], torch.tensor([matched_query_id_offset]).to(device='cuda'))
       
        return loss_ce, loss_peak

    def loss_bboxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][src_idx]
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes.clip(min=0)),
            box_cxcyczwhd_to_xyzxyz(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def _get_src_permutation_idx(self, matches):
        batch_idx = []
        src_idx = []
        for idx, batch_matches in enumerate(matches):
            for match in batch_matches:
                batch_idx.append(idx)
                src_idx.append(match[1])

        return torch.tensor(batch_idx), torch.tensor(src_idx)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # Compute losses
        loss_bbox, loss_giou = self.loss_bboxes(outputs, targets, indices, num_boxes)
        loss_cls, loss_peak = self.loss_class(outputs, targets, indices)

        loss_dict = {
            'bbox': loss_bbox,
            'giou': loss_giou,
            'cls': loss_cls,
            'peak': loss_peak
        }

        # Compute losses for the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)

                loss_bbox, loss_giou = self.loss_bboxes(aux_outputs, targets, indices, num_boxes)
                loss_cls, loss_peak = self.loss_class(aux_outputs, targets, indices)

                loss_dict[f'bbox_{i}'] = loss_bbox
                loss_dict[f'giou_{i}'] = loss_giou
                loss_dict[f'cls_{i}'] = loss_cls
                loss_dict[f'peak_{i}'] = loss_peak

        return loss_dict

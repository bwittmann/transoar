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
        # self.cls_weights = torch.tensor(
        #     [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        # ).type(torch.FloatTensor)
        self.cls_weights = torch.tensor([1, 81]).type(torch.FloatTensor)


    def loss_class(self, outputs, soft_labels):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        cls_preds = outputs['pred_logits'].flatten()
        cls_labels = soft_labels.flatten()

        # Remove non existent classes
        valid_ids = (cls_labels != -1).nonzero()

        loss_ce = F.binary_cross_entropy_with_logits(cls_preds[valid_ids].squeeze(), cls_labels[valid_ids].squeeze())
        return loss_ce

    def loss_bboxes(self, outputs, targets, matches, num_boxes, matches_per_class=1):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        bs = outputs['pred_boxes'].shape[0]
        box_preds = outputs['pred_boxes'].reshape(bs, 20, 27, -1)
        box_labels = [target['boxes'] for target in targets]    # can have different shapes

        # Get matched pred boxes
        match_ids = matches.nonzero().T.unbind()
        matched_box_preds = box_preds[match_ids]
        matched_box_labels = torch.cat([torch.repeat_interleave(labels, matches_per_class, dim=0) for labels in box_labels])    # TODO: make smarter choices
        
        # Determine bbox losses        
        loss_bbox = F.l1_loss(matched_box_preds, matched_box_labels, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes * matches_per_class

        loss_giou = 1 - torch.diag(generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(matched_box_preds.clip(min=0)),
            box_cxcyczwhd_to_xyzxyz(matched_box_labels))
        )
        loss_giou = loss_giou.sum() / num_boxes * matches_per_class

        return loss_bbox, loss_giou



    def forward(self, outputs, targets, anchors):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        matches, soft_labels = self.matcher(outputs, targets, anchors)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # Compute losses
        loss_cls = self.loss_class(outputs, soft_labels)
        loss_bbox, loss_giou = self.loss_bboxes(outputs, targets, matches, num_boxes)


        loss_dict = {
            'bbox': loss_bbox,
            'giou': loss_giou,
            'cls': loss_cls
        }

        # Compute losses for the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                matches, soft_labels = self.matcher(aux_outputs, targets, anchors)

                loss_cls = self.loss_class(aux_outputs, soft_labels)
                loss_bbox, loss_giou = self.loss_bboxes(aux_outputs, targets, matches, num_boxes)

                loss_dict[f'bbox_{i}'] = loss_bbox
                loss_dict[f'giou_{i}'] = loss_giou
                loss_dict[f'cls_{i}'] = loss_cls

        return loss_dict

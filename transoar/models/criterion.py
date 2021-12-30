"""Module containing the loss functions of the transoar project."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.utils.bboxes import generalized_bbox_iou_3d, box_cxcyczwhd_to_xyzxyz

class TransoarCriterion(nn.Module):
    """This class computes the loss for TransoarNet."""
    def __init__(self):
        """Create the criterion."""
        super().__init__()

    def _loss_bboxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        target_boxes_stacked, output_boxes_stacked, output_classes, output_boxes = self._prepare_data(targets, outputs)

        loss_bbox = F.l1_loss(output_boxes_stacked, target_boxes_stacked, reduction='none')

        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(output_boxes_stacked),
            box_cxcyczwhd_to_xyzxyz(target_boxes_stacked))
        )
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou, output_boxes, output_classes

    @staticmethod
    def _prepare_data(targets, outputs):
        target_boxes_stacked = torch.cat([t['boxes']for t in targets], dim=0)
        target_indices = [t['labels'] - 1 for t in targets] # To cope with noise in dataset labels
        output_boxes = [o[idx] for o, idx in zip(outputs['pred_boxes'], target_indices)]
        output_boxes_stacked = torch.cat(output_boxes)
        
        assert target_boxes_stacked.shape == output_boxes_stacked.shape
        return target_boxes_stacked, output_boxes_stacked, target_indices, output_boxes

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # Compute losses and additionally get adjusted output classes due to noisy dataset labels
        loss_bbox, loss_giou, output_boxes, output_classes = self._loss_bboxes(outputs, targets, num_boxes)

        loss_dict = {
            'bbox': loss_bbox,
            'giou': loss_giou
        }

        # Compute losses for the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                loss_bbox, loss_giou, _, _ = self._loss_bboxes(aux_outputs, targets, num_boxes)
                loss_dict[f'bbox_{i}'] = loss_bbox
                loss_dict[f'giou_{i}'] = loss_giou

        return loss_dict, output_boxes, output_classes

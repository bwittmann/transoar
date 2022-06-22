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
    def __init__(self, num_classes, matcher, seg_proxy, seg_fg_bg):
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

        self._seg_proxy = seg_proxy
        self._seg_fg_bg = seg_fg_bg

        if seg_proxy:
            self._dice_loss = SoftDiceLoss(
                nonlin=torch.nn.Softmax(dim=1), batch_dice=True, smooth_nom=1e-05, smooth_denom=1e-05,do_bg=False
            )

        # Hack to make deterministic, https://github.com/pytorch/pytorch/issues/46024
        self.cls_weights = torch.tensor(
            [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        ).type(torch.FloatTensor)

    def loss_class(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.cls_weights.to(device=src_logits.device))

        return loss_ce

    def loss_bboxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes))
        )
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou

    def loss_segmentation(self, outputs, targets):
        assert 'pred_seg' in outputs

        # Get only fg and bg labels
        if self._seg_fg_bg:
           targets[targets > 0] = 1
        targets = targets.squeeze(1).long()

        # Determine segmentatio losses
        loss_ce = F.cross_entropy(outputs['pred_seg'], targets)
        loss_dice = self._dice_loss(outputs['pred_seg'], targets)
        
        return loss_ce, loss_dice

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, seg_targets):
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
        loss_cls = self.loss_class(outputs, targets, indices)

        if self._seg_proxy:
            loss_seg_ce, loss_seg_dice = self.loss_segmentation(outputs, seg_targets)

        loss_dict = {
            'bbox': loss_bbox,
            'giou': loss_giou,
            'cls': loss_cls,
            'segce': loss_seg_ce if self._seg_proxy else torch.tensor(0),
            'segdice': loss_seg_dice if self._seg_proxy else torch.tensor(0)
        }

        # Compute losses for the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)

                loss_bbox, loss_giou = self.loss_bboxes(aux_outputs, targets, indices, num_boxes)
                loss_cls = self.loss_class(aux_outputs, targets, indices)

                loss_dict[f'bbox_{i}'] = loss_bbox
                loss_dict[f'giou_{i}'] = loss_giou
                loss_dict[f'cls_{i}'] = loss_cls

        return loss_dict

class SoftDiceLoss(nn.Module):
    def __init__(
        self, nonlin=None, batch_dice=False, do_bg=False, 
        smooth_nom=1e-5, smooth_denom=1e-5
    ):
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.nonlin = nonlin
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom

    def forward(self, inp, target, loss_mask=None):
        shp_x = inp.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.nonlin is not None:
            inp = self.nonlin(inp)

        tp, fp, fn = get_tp_fp_fn(inp, target, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth_nom
        denominator = 2 * tp + fp + fn + self.smooth_denom

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    return tp, fp, fn

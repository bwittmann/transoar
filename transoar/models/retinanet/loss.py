import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class BCEWithLogitsLossOneHot(nn.BCEWithLogitsLoss):
    def __init__(self,
                 *args,
                 num_classes: int,
                 smoothing: float = 0.0,
                 loss_weight: float = 1.,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self._smoothing = smoothing
        self._num_classes = num_classes
        self._loss_weight = loss_weight

    @staticmethod
    def _one_hot_smooth(data, num_classes, smoothing):
        targets = torch.empty(size=(*data.shape, num_classes), device=data.device)\
            .fill_(smoothing / num_classes)\
            .scatter_(-1, data.long().unsqueeze(-1), 1. - smoothing)
        return targets

    def forward(self, input, target):
        target_one_hot = self._one_hot_smooth(
            target, num_classes=self._num_classes + 1, smoothing=self._smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded

        return self._loss_weight * super().forward(input, target_one_hot.float())

class GIoULoss(nn.Module):
    def __init__(self, eps=1e-7, loss_weight=1.):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pred_boxes, target_boxes):
        loss = torch.sum(
            torch.diag(generalized_box_iou(pred_boxes, target_boxes, eps=self.eps), # (x1, y1, x2, y2, (z1, z2)) [N, dim * 2]
                       diagonal=0)
                )
        return self.loss_weight * -1 * loss

@autocast(enabled=False)
def generalized_box_iou(boxes1, boxes2, eps=0):
    if boxes1.nelement() == 0 or boxes2.nelement() == 0:
        return torch.tensor([]).to(boxes1)
    else:
        return generalized_box_iou_3d(boxes1.float(), boxes2.float(), eps=eps)

def generalized_box_iou_3d(boxes1, boxes2, eps=0) :
    iou, union = box_iou_union_3d(boxes1, boxes2)

    x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.min(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.max(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    vol = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    return iou - (vol - union) / vol

def box_iou_union_3d(boxes1, boxes2, eps=0):
    vol1 = box_area_3d(boxes1)
    vol2 = box_area_3d(boxes2)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.max(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.min(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    union = (vol1[:, None] + vol2 - inter)
    return inter / union, union

def box_area_3d(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])

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
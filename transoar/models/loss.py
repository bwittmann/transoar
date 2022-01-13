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
            target, num_classes=self.num_classes + 1, smoothing=self.smoothing)  # [N, C + 1]
        target_one_hot = target_one_hot[:, 1:]  # background is implicitly encoded

        return self.loss_weight * super().forward(input, target_one_hot.float())


class GIoULoss(torch.nn.Module):
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
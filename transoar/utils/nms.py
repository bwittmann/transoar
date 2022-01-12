import torch

def nms_cpu(boxes, scores, thresh):
    """
    Performs non-maximum suppression for 3d boxes on cpu
    
    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]    TODO
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded
    
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    ious, _ = box_iou(boxes, boxes)
    _, _idx = torch.sort(scores, descending=True)
    keep = []
    while _idx.nelement() > 0:
        keep.append(_idx[0])
        # get all elements that were not matched and discard all others.
        non_matches = torch.where((ious[_idx[0]][_idx] <= thresh))[0]
        _idx = _idx[non_matches]
    return torch.tensor(keep).to(boxes).long()


def box_iou(boxes1, boxes2, eps: float = 0) :
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.
    
    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        Tensor[N, M]: the nxM matrix containing the pairwise union
            values
    """
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
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2, z1, z2) coordinates.
    
    Arguments:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
    Returns:
        area (Union[Tensor, ndarray]): area for each box [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])
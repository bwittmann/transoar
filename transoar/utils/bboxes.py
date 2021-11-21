"""Helper functions for handling bounding boxes."""

import torch


def generalized_bbox_iou_3d(bboxes1, bboxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, z0, x1, y1. z1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (bboxes1[:, 3:] >= bboxes1[:, :3]).all()
    assert (bboxes2[:, 3:] >= bboxes2[:, :3]).all()
    iou, union = iou_3d(bboxes1, bboxes2)

    x1 = torch.min(bboxes1[:, None, 0], bboxes2[:, 0])
    y1 = torch.min(bboxes1[:, None, 1], bboxes2[:, 1])
    z1 = torch.min(bboxes1[:, None, 2], bboxes2[:, 2])
    x2 = torch.max(bboxes1[:, None, 3], bboxes2[:, 3])
    y2 = torch.max(bboxes1[:, None, 4], bboxes2[:, 4])
    z2 = torch.max(bboxes1[:, None, 5], bboxes2[:, 5])

    dx = (x2 - x1).clamp(min=0)
    dy = (y2 - y1).clamp(min=0)
    dz = (z2 - z1).clamp(min=0)

    vol = dx * dy * dz
    return iou - (vol - union) / vol

def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]

    return torch.stack(b, dim=-1)

def segmentation2bbox(segmentation_maps, padding, box_format='cxcyczwhd', normalize=True):
    batch_bboxes = []
    batch_classes = []
    for map_ in segmentation_maps:
        assert map_.ndim == 4

        bboxes = []
        classes = [int(class_) for class_ in map_.unique(sorted=True)][1:]
        batch_classes.append(classes)

        for class_ in classes:
            class_indices = (map_ == class_).nonzero(as_tuple=False)

            min_values = class_indices.min(dim=0)[0][1:].to(torch.float)   # x, y, z
            max_values = class_indices.max(dim=0)[0][1:].to(torch.float)

            # Apply padding to bounding boxes
            min_values -= padding
            max_values += padding

            assert min_values[0] < max_values[0]
            assert min_values[1] < max_values[1]
            assert min_values[2] < max_values[2]

            if normalize:   # Put coords between 0 and 1; nec for sigmoid output
                min_values /= torch.tensor(map_.shape[1:])
                max_values /= torch.tensor(map_.shape[1:])

            if box_format == 'xyzxyz':
                bboxes.append(torch.hstack((min_values, max_values)))   # x1, y1, z1, x2, y2, z2
            elif box_format == 'cxcyczwhd':
                width, height, depth = max_values - min_values
                cx, cy, cz = (max_values + min_values) / 2
                bboxes.append(torch.tensor([cx, cy, cz, width, height, depth]))
            else:
                raise ValueError('Please select a valid box format.')

        batch_bboxes.append(torch.vstack(bboxes))

    return batch_bboxes, batch_classes

def iou_3d(bboxes1, bboxes2):
    """Determines the intersection over union (IoU) for two sets of
    three dimensional bounding boxes.

    Bounding boxes have to be in the format (x1, y1, z1, x2, y2, z2).

    Args:
        bboxes1: A tensor of the shape [N, 6] containing the first
            set of bounding boxes.
        bboxes2: A tensor of the shape [M, 6] containing the first
            set of bounding boxes.

    Returns:
        A tensor of shape [N, M] containing the IoU values of all 
        bounding boxes to each other and a tensor of same shape containing
        the pure union values between bboxes. Keep in mind that the diagonal
        consists of ones.
    """
    volume_bbox1 = bboxes_volume(bboxes1)
    volume_bbox2 = bboxes_volume(bboxes2)

    x1 = torch.max(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = torch.max(bboxes1[:, None, 1], bboxes2[None, :, 1])
    z1 = torch.max(bboxes1[:, None, 2], bboxes2[None, :, 2])
    x2 = torch.min(bboxes1[:, None, 3], bboxes2[None, :, 3])
    y2 = torch.min(bboxes1[:, None, 4], bboxes2[None, :, 4])
    z2 = torch.min(bboxes1[:, None, 5], bboxes2[None, :, 5])

    delta_x = (x2 - x1).clamp(min=0)
    delta_y = (y2 - y1).clamp(min=0)
    delta_z = (z2 - z1).clamp(min=0)

    intersection = delta_x * delta_y * delta_z
    union = volume_bbox1[:, None] + volume_bbox2 - intersection
    iou = intersection / union

    return iou, union 

def bboxes_volume(bboxes):
    """Estimates the volume of a three dimensional bounding box.
    
    Args:
        bboxes: A tensor of the shape [N, 6] containing N bounding
            boxes in the format (x1, y1, z1, x2, y2, z2).

    Returns:
        A tensor of shape (N,) containing the corresponding volumes.
    """
    delta_x = bboxes[:, 3] - bboxes[:, 0]
    delta_y = bboxes[:, 4] - bboxes[:, 1]
    delta_z = bboxes[:, 5] - bboxes[:, 2]
    return delta_x * delta_y * delta_z
import torch
from torch.autograd import Variable
import numpy as np

def gt_anchor_matching(config, anchors, gt_boxes, gt_class_ids=None):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): [num_gt_boxes] Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)

    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral.
               In case of one stage detectors like RetinaNet/RetinaUNet this flag takes
               class_ids as positive anchor values, i.e. values >= 1!
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    """

    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((config['train_anchors_per_image'], 2*config['dim']))
    anchor_matching_iou = config['anchor_matching_iou']

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    anchor_class_matches[(anchor_iou_max < 0.01)] = -1

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    # 3. Set anchors with high overlap as positive.
    above_tresh_ixs = np.argwhere(anchor_iou_max >= anchor_matching_iou)
    anchor_class_matches[above_tresh_ixs] = gt_class_ids[anchor_iou_argmax[above_tresh_ixs]]

    # Subsample to balance positive anchors.
    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (config['train_anchors_per_image'] // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    # Leave all negative proposals negative now and sample from them in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0  # index into anchor_delta_targets
    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if config['dim'] == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]

        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d

            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)
            ]

        # normalize.
        anchor_delta_targets[ix] /= config['bbox_std_dev']
        ix += 1

    return anchor_class_matches, anchor_delta_targets


def generate_pyramid_anchors(config):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    from configs:
    :param scales: cf.RPN_ANCHOR_SCALES , e.g. [4, 8, 16, 32]
    :param ratios: cf.RPN_ANCHOR_RATIOS , e.g. [0.5, 1, 2]
    :param feature_shapes: cf.BACKBONE_SHAPES , e.g.  [array of shapes per feature map] [80, 40, 20, 10, 5]
    :param feature_strides: cf.BACKBONE_STRIDES , e.g. [2, 4, 8, 16, 32, 64]
    :param anchors_stride: cf.RPN_ANCHOR_STRIDE , e.g. 1
    :return anchors: (N, (y1, x1, y2, x2, (z1), (z2)). All generated anchors in one array. Sorted
    with the same order of the given scales. So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    """
    scales = config['anchor_scales']
    ratios = config['anchor_ratios']
    feature_shapes = config['backbone_shapes']
    anchor_stride = config['anchor_stride']
    pyramid_levels = config['pyramid_levels']
    feature_strides = config['backbone_strides']

    anchors = []
    print("feature map shapes: {}".format(feature_shapes))
    print("anchor scales: {}".format(scales))

    expected_anchors = [np.prod(feature_shapes[ii]) * len(ratios) * len(scales['xy'][ii]) for ii in pyramid_levels]

    for lix, level in enumerate(pyramid_levels):
        anchors.append(
            generate_anchors_3D(
                scales['xy'][level], scales['z'][level], ratios, feature_shapes[level],
                feature_strides['xy'][level], feature_strides['z'][level], anchor_stride
            )
        )

        print("level {}: built anchors {} / expected anchors {} ||| total build {} / total expected {}".format(
            level, anchors[-1].shape, expected_anchors[lix], np.concatenate(anchors).shape, np.sum(expected_anchors)))

    out_anchors = np.concatenate(anchors, axis=0)
    return out_anchors

def generate_anchors_3D(scales_xy, scales_z, ratios, shape, feature_stride_xy, feature_stride_z, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales_xy, ratios_meshed = np.meshgrid(np.array(scales_xy), np.array(ratios))
    scales_xy = scales_xy.flatten()
    ratios_meshed = ratios_meshed.flatten()

    # Enumerate heights and widths from scales and ratios - same for each cell in feature map
    heights = scales_xy / np.sqrt(ratios_meshed)
    widths = scales_xy * np.sqrt(ratios_meshed)
    depths = np.tile(np.array(scales_z), len(ratios_meshed)//np.array(scales_z)[..., None].shape[0])

    # Enumerate shifts in feature space - to get to original image coords
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride_xy #translate from fm positions to input coords.
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride_xy
    shifts_z = np.arange(0, shape[2], anchor_stride) * (feature_stride_z)
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    return boxes

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    volume1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 4])
    volume2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 4])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]  # this is the gt box
        overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
    return overlaps

def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2, z1, z2] (typically gt box)
    boxes: [boxes_count, (y1, x1, y2, x2, z1, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou

def clip_boxes_numpy(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2 / [N, 6] in 3D.
    window: iamge shape (y, x, (z))
    """
    if boxes.shape[1] == 4:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
            np.clip(boxes[:, 1], 0, window[0])[:, None],
            np.clip(boxes[:, 2], 0, window[1])[:, None],
            np.clip(boxes[:, 3], 0, window[1])[:, None]), 1
        )

    else:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
             np.clip(boxes[:, 1], 0, window[0])[:, None],
             np.clip(boxes[:, 2], 0, window[1])[:, None],
             np.clip(boxes[:, 3], 0, window[1])[:, None],
             np.clip(boxes[:, 4], 0, window[2])[:, None],
             np.clip(boxes[:, 5], 0, window[2])[:, None]), 1
        )

    return boxes


def shem(roi_probs_neg, negative_count, ohem_poolsize):
    """
    stochastic hard example mining: from a list of indices (referring to non-matched predictions),
    determine a pool of highest scoring (worst false positives) of size negative_count*ohem_poolsize.
    Then, sample n (= negative_count) predictions of this pool as negative examples for loss.
    :param roi_probs_neg: tensor of shape (n_predictions, n_classes).
    :param negative_count: int.
    :param ohem_poolsize: int.
    :return: (negative_count).  indices refer to the positions in roi_probs_neg. If pool smaller than expected due to
    limited negative proposals availabel, this function will return sampled indices of number < negative_count without
    throwing an error.
    """
    # sort according to higehst foreground score.
    probs, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = torch.tensor((ohem_poolsize * int(negative_count), order.size()[0])).min().int()
    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0])
    return pool_indices[rand_idx[:negative_count].cuda()]

def apply_box_deltas_3D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is y1, x1, y2, x2, z1, z2
    deltas: [N, 6] where each row is [dy, dx, dz, log(dh), log(dw), log(dd)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 4]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 4] + 0.5 * depth
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= torch.exp(deltas[:, 3])
    width *= torch.exp(deltas[:, 4])
    depth *= torch.exp(deltas[:, 5])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    result = torch.stack([y1, x1, y2, x2, z1, z2], dim=1)
    return result

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2) / 3D: (z1, z2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]  / 3D: (z1, z2)
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    if boxes.shape[1] > 5:
        boxes[:, 4] = boxes[:, 4].clamp(float(window[4]), float(window[5]))
        boxes[:, 5] = boxes[:, 5].clamp(float(window[4]), float(window[5]))

    return boxes

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]
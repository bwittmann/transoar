import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from transoar.models.backbones.fpn import FPN, NDConvGenerator
from transoar.utils.anchors import *
from transoar.utils.nms import nms_cpu

class Classifier(nn.Module):
    def __init__(self, config, conv):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.n_classes = config['head_classes']
        n_input_channels = config['end_filts']
        n_features = config['n_features']
        n_output_channels = config['n_anchors_per_pos'] * config['head_classes']
        anchor_stride = config['anchor_stride']
        relu = config['relu']

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride, pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]

class BBRegressor(nn.Module):
    def __init__(self, config, conv):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = conv.dim
        n_input_channels = config['end_filts']
        n_features = config['n_features']
        n_output_channels = config['n_anchors_per_pos'] * self.dim * 2
        anchor_stride = config['anchor_stride']
        relu = config['relu']

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=relu)
        self.conv_final = conv(n_features, n_output_channels, ks=3, stride=anchor_stride,
                               pad=1, relu=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix

def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def refine_detections(anchors, probs, deltas, batch_ixs, config):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    anchors = anchors.repeat(len(batch_ixs.unique()), 1)

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:config['pre_nms_limit']]
    # reshape indices to 2D index array with shape like fg_probs - 0: proposal id, 1: class
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1).to(dtype=torch.int), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    pre_nms_scores = flat_probs[:config['pre_nms_limit']]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background class again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long().cuda()

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(config['bbox_std_dev'], [1, config['dim'] * 2])).float().cuda()
    scale = torch.from_numpy(np.array(config['scale'])).float().cuda()
    refined_rois = apply_box_deltas_3D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = clip_to_window(config['window'], refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]

            class_keep = nms_cpu(ix_rois, ix_scores, float(config['det_nms_thres']))

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:config['model_max_instances_per_batch_element']]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result

def get_results(config, img_shape, detections, seg_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, config['dim']*2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            boxes = detections[ix][:, :2 * config['dim']].astype(np.int32)
            class_ids = detections[ix][:, 2 * config['dim'] + 1].astype(np.int32)
            scores = detections[ix][:, 2 * config['dim'] + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if config['dim'] == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= config['model_min_confidence']:
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_pred_class_id': class_ids[ix2]})

    return box_results_list


class RetinaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build()

    def build(self):
        """
        Build Retina Net architecture.
        """
        self.fpn = FPN(self.config)
        self.np_anchors = generate_pyramid_anchors(self.config)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()

        self.classifier = Classifier(self.config, NDConvGenerator(dim=3))
        self.bb_regressor = BBRegressor(self.config, NDConvGenerator(dim=3))


    def train_forward(self, data, gt_boxes, gt_class_ids, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = data

        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas, seg_logits = self.forward(img)

        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = gt_anchor_matching(
                    self.config, self.np_anchors, gt_boxes[b], gt_class_ids[b])

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            class_loss, _ = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]

        box_results = get_results(self.config, img.shape, detections, seg_logits, box_results_list)
        loss = batch_class_loss + batch_bbox_loss

        return loss, box_results

    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        detections, _, _, seg_logits = self.forward(img)
        results_dict = get_results(self.config, img.shape, detections, seg_logits)
        return results_dict


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.fpn(img)
        seg_logits = None
        selected_fmaps = [fpn_outs[i] for i in self.config['pyramid_levels']]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.classifier(p))
            bb_reg_layer_outputs.append(self.bb_regressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

        # merge batch_dimension and store info in batch_ixs for re-allocation.
        batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
        flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, self.config)

        return detections, class_logits, bb_outputs, seg_logits

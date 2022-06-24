"""
Module containing the evaluator of the transoar project.
Parts are adapted from https://github.com/cocodataset/cocoapi and https://github.com/MIC-DKFZ/nnDetection.
"""

from functools import partial

import numpy as np

from transoar.metric import Metric
from transoar.utils.bboxes import iou_3d_np


class DetectionEvaluator:
    def __init__(
        self,
        classes,
        classes_small,
        classes_mid,
        classes_large,
        iou_range_coco,
        iou_range_nndet,
        iou_fn=iou_3d_np,
        max_detections=1,
        sparse_results=False
    ):
        """
        Class for evaluate detection metrics

        Args:
            metrics (Sequence[DetectionMetric]: detection metrics to evaluate
            iou_fn (Callable[[np.ndarray, np.ndarray], np.ndarray]): compute overlap for each pair
            max_detections (int): number of maximum detections per image (reduces computation)
        """
        self.iou_fn = iou_fn
        self.max_detections = max_detections

        self.results_list = []  # store results of each image

        self.metrics = [
            Metric(
                classes=classes,
                classes_small=classes_small,
                classes_mid=classes_mid,
                classes_large=classes_large,
                iou_list=(0.1, 0.5, 0.75), #np.arange(0.1, 1.0, 0.1),  # for individual APs
                iou_range_coco=iou_range_coco, # for mAP
                iou_range_nndet=iou_range_nndet,
                per_class=False if sparse_results else True,
                determine_ar=False,
                max_detection=(1, ) # different from nndet (100, )
            )
        ]

        self.iou_thresholds = self.get_unique_iou_thresholds()
        self.iou_mapping = self.get_indices_of_iou_for_each_metric()

    def get_unique_iou_thresholds(self):
        """
        Compute unique set of iou thresholds
        """
        iou_thresholds = [_i for i in self.metrics for _i in i.get_iou_thresholds()]
        iou_thresholds = list(set(iou_thresholds))
        iou_thresholds.sort()
        return iou_thresholds

    def get_indices_of_iou_for_each_metric(self):
        """
        Find indices of iou thresholds for each metric
        """
        return [[self.iou_thresholds.index(th) for th in m.get_iou_thresholds()]
                for m in self.metrics]

    def add(
        self,
        pred_boxes,
        pred_classes,
        pred_scores,
        gt_boxes,
        gt_classes,
        gt_ignore=None
    ):
        """
        Preprocess batch results for final evaluation

        Args:
            pred_boxes (Sequence[np.ndarray]): predicted boxes from single batch; List[[D, dim * 2]], D number of
                predictions
            pred_classes (Sequence[np.ndarray]): predicted classes from a single batch; List[[D]], D number of
                predictions
            pred_scores (Sequence[np.ndarray]): predicted score for each bounding box; List[[D]], D number of
                predictions
            gt_boxes (Sequence[np.ndarray]): ground truth boxes; List[[G, dim * 2]], G number of ground truth
            gt_classes (Sequence[np.ndarray]): ground truth classes; List[[G]], G number of ground truth
            gt_ignore (Sequence[Sequence[bool]]): specified if which ground truth boxes are not counted as true
                positives (detections which match theses boxes are not counted as false positives either);
                List[[G]], G number of ground truth

        Returns
            dict: empty dict... detection metrics can only be evaluated at the end
        """
        # reduce class ids by 1 to start with 0
        gt_classes = [batch_elem_classes -1 for batch_elem_classes in gt_classes]
        pred_classes = [batch_elem_classes -1 for batch_elem_classes in pred_classes]

        if gt_ignore is None:   # only zeros -> don't ignore anything
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]

        self.results_list.extend(matching_batch(
            self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore,
            max_detections=self.max_detections))

        return {}

    def eval(self):
        """
        Accumulate results of individual batches and compute final metrics

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(self.iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = list(map(_filter, self.results_list))    # no filtering
            
            score, curve = metric(iou_filtered_results)
            
            if score is not None:
                metric_scores.update(score)
            
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores

    @staticmethod
    def iou_filter(image_dict, iou_idx,
                   filter_keys=('dtMatches', 'gtMatches', 'dtIgnore')):
        """
        This functions can be used to filter specific IoU values from the results
        to make sure that the correct IoUs are passed to metric
        
        Parameters
        ----------
        image_dict : dict
            dictionary containin :param:`filter_keys` which contains IoUs in the first dimension
        iou_idx : List[int]
            indices of IoU values to filter from keys
        filter_keys : tuple, optional
            keys to filter, by default ('dtMatches', 'gtMatches', 'dtIgnore')
        
        Returns
        -------
        dict
            filtered dictionary
        """
        iou_idx = list(iou_idx)
        filtered = {}
        for cls_key, cls_item in image_dict.items():
            filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item
                                 for key, item in cls_item.items()}
        return filtered

    def reset(self):
        """
        Reset internal state of evaluator
        """
        self.results_list = []


def matching_batch(
    iou_fn, 
    iou_thresholds, 
    pred_boxes,
    pred_classes, 
    pred_scores,
    gt_boxes, 
    gt_classes,
    gt_ignore,
    max_detections
):
    """
    Match boxes of a batch to corresponding ground truth for each category
    independently.

    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives
            (detections which match theses boxes are not counted as false
            positives either); List[[G]], G number of ground truth
        max_detections: maximum number of detections which should be evaluated

    Returns:
        List[Dict[int, Dict[str, np.ndarray]]]
            matched detections [dtMatches] and ground truth [gtMatches]
            boxes [str, np.ndarray] for each category (stored in dict keys)
            for each image (list)
    """
    results = []
    # iterate over images/batches
    for pboxes, pclasses, pscores, gboxes, gclasses, gignore in zip(
        pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore
    ):
        img_classes = np.union1d(pclasses, gclasses)
        result = {}  # dict contains results for each class in one image
        for c in img_classes:
            pred_mask = pclasses == c # mask predictions with current class
            gt_mask = gclasses == c # mask ground trtuh with current class

            if not np.any(gt_mask): # no ground truth
                result[c] = _matching_no_gt(
                    iou_thresholds=iou_thresholds,
                    pred_scores=pscores[pred_mask],
                    max_detections=max_detections)
            elif not np.any(pred_mask): # no predictions
                result[c] = _matching_no_pred(
                    iou_thresholds=iou_thresholds,
                    gt_ignore=gignore[gt_mask],
                )
            else: # at least one prediction and one ground truth
                result[c] = _matching_single_image_single_class(
                    iou_fn=iou_fn,
                    pred_boxes=pboxes[pred_mask],
                    pred_scores=pscores[pred_mask],
                    gt_boxes=gboxes[gt_mask],
                    gt_ignore=gignore[gt_mask],
                    max_detections=max_detections,
                    iou_thresholds=iou_thresholds,
                )
        results.append(result)
    return results


def _matching_no_gt(
    iou_thresholds,
    pred_scores,
    max_detections,
):
    """
    Matching result with not ground truth in image

    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        dt_scores: predicted scores
        max_detections: maximum number of allowed detections per image.
            This functions uses this parameter to stay consistent with
            the actual matching function which needs this limit.

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]
    dt_scores = pred_scores[dt_ind]

    num_preds = len(dt_scores)

    gt_match = np.array([[]] * len(iou_thresholds))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': np.array([]).reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_no_pred(
    iou_thresholds,
    gt_ignore,
):
    """
    Matching result with no predictions

    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_scores = np.array([])
    dt_match = np.array([[]] * len(iou_thresholds))
    dt_ignore = np.array([[]] * len(iou_thresholds))

    n_gt = 0 if gt_ignore.size == 0 else gt_ignore.shape[0]
    gt_match = np.zeros((len(iou_thresholds), n_gt))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_single_image_single_class(
    iou_fn,
    pred_boxes,
    pred_scores,
    gt_boxes,
    gt_ignore,
    max_detections,
    iou_thresholds,    
):
    """
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; [D, dim * 2], D number
            of predictions
        pred_scores: predicted score for each bounding box; [D], D number of
            predictions
        gt_boxes: ground truth boxes; [G, dim * 2], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth
        max_detections: maximum number of detections which should be evaluated

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    # filter for max_detections highest scoring predictions to speed up computation
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]    # only take up to max number of detections

    pred_boxes = pred_boxes[dt_ind] # sort by highest score
    pred_scores = pred_scores[dt_ind]

    # sort ignored ground truth to last positions
    gt_ind = np.argsort(gt_ignore, kind='mergesort')
    gt_boxes = gt_boxes[gt_ind]
    gt_ignore = gt_ignore[gt_ind]

    # ious between sorted(!) predictions and ground truth
    ious = iou_fn(pred_boxes, gt_boxes)

    num_preds, num_gts = ious.shape[0], ious.shape[1]
    gt_match = np.zeros((len(iou_thresholds), num_gts))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    for tind, t in enumerate(iou_thresholds):
        for dind, _d in enumerate(pred_boxes):  # iterate detections starting from highest scoring one
            # information about best match so far (m=-1 -> unmatched)
            iou = min([t, 1-1e-10]) # iou threshold
            m = -1

            for gind, _g in enumerate(gt_boxes):  # iterate ground truth
                # if this gt already matched, continue (no duplicate detections)
                if gt_match[tind, gind] > 0:
                    continue

                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                    break

                # continue to next gt unless better match made
                if ious[dind, gind] < iou:
                    continue

                # if match successful and best so far, store appropriately
                iou = ious[dind, gind]
                m = gind

            # if match made, store id of match for both dt and gt
            if m == -1:
                continue
            else:
                dt_ignore[tind, dind] = int(gt_ignore[m])
                dt_match[tind, dind] = 1
                gt_match[tind, m] = 1

    # store results for given image and category
    return {
            'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
            'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
            'dtScores': pred_scores,  # [D] detection scores
            'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
            'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
        }

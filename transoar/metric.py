"""
Module containing the metric of the transoar project.
Parts are adapted from https://github.com/cocodataset/cocoapi and https://github.com/MIC-DKFZ/nnDetection.
"""

import numpy as np

class Metric:
    def __init__(
        self,
        classes,
        iou_list=(0.1, 0.5, 0.75),
        iou_range=(0.1, 0.5, 0.05),
        max_detection=(1, 5, 100),
        per_class=True
    ):
        """
        Class to compute COCO metrics
        Metrics computed:
            mAP over the IoU range specified by :param:`iou_range` at last value of :param:`max_detection`
            AP values at IoU thresholds specified by :param:`iou_list` at last value of :param:`max_detection`
            AR over max detections thresholds defined by :param:`max_detection` (over iou range)

        Args:
            classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
            iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
            iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
            max_detection (Sequence[int]): maximum number of detections per image
        """
        self.classes = classes
        self.per_class = per_class

        iou_list = np.array(iou_list)
        _iou_range = np.linspace(iou_range[0], iou_range[1],
                                 int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True)
        self.iou_thresholds = np.union1d(iou_list, _iou_range)
        self.iou_range = iou_range

        # get indices of iou values of ious range and ious list for later evaluation
        self.iou_list_idx = np.nonzero(iou_list[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        self.iou_range_idx = np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]

        assert (self.iou_thresholds[self.iou_list_idx] == iou_list).all()
        assert (self.iou_thresholds[self.iou_range_idx] == _iou_range).all()

        self.recall_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.max_detections = max_detection

    def get_iou_thresholds(self):
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(
        self,
        results_list
    ):
        """
        Compute COCO metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of 
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored

        Returns:
            Dict[str, float]: dictionary with coco metrics
            Dict[str, np.ndarray]: None
        """
        dataset_statistics = self.compute_statistics(results_list=results_list)

        results = {}
        results.update(self.compute_ap(dataset_statistics))
        results.update(self.compute_ar(dataset_statistics))

        return results, None

    def compute_ap(self, dataset_statistics):
        """
        Compute AP metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        if self.iou_range:  # mAP
            key = (f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                   f"MaxDet_{self.max_detections[-1]}")
            results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx,
                                                  cls_idx=cls_idx, max_det_idx=-1)

        for idx in self.iou_list_idx:   # AP@IoU
            key = f"AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ap(dataset_statistics, iou_idx=[idx], max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AP_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics,
                                                  iou_idx=[idx], cls_idx=cls_idx, max_det_idx=-1)
        return results

    def compute_ar(self, dataset_statistics):
        """
        Compute AR metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        for max_det_idx, max_det in enumerate(self.max_detections):  # mAR
            key = f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{max_det}"
            results[key] = self.select_ar(dataset_statistics, max_det_idx=max_det_idx)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                           f"MaxDet_{max_det}")
                    results[key] = self.select_ar(dataset_statistics,
                                                  cls_idx=cls_idx, max_det_idx=max_det_idx)

        for idx in self.iou_list_idx:   # AR@IoU
            key = f"AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ar(dataset_statistics, iou_idx=idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AR_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ar(dataset_statistics, iou_idx=idx,
                                                  cls_idx=cls_idx, max_det_idx=-1)
        return results

    @staticmethod
    def select_ap(
        dataset_statistics,
        iou_idx=None,
        cls_idx=None,
        max_det_idx=-1
    ):
        """
        Compute average precision

        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: AP value
        """
        prec = dataset_statistics["precision"]
        if iou_idx is not None:
            prec = prec[iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx, :]
        prec = prec[..., max_det_idx]
        return np.mean(prec)

    @staticmethod
    def select_ar(
        dataset_statistics,
        iou_idx=None,
        cls_idx=None,
        max_det_idx=-1
    ):
        """
        Compute average recall

        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: recall value
        """
        rec = dataset_statistics["recall"]
        if iou_idx is not None:
            rec = rec[iou_idx]
        if cls_idx is not None:
            rec = rec[..., cls_idx, :]
        rec = rec[..., max_det_idx]

        if len(rec[rec > -1]) == 0:
            rec = -1
        else:
            rec = np.mean(rec[rec > -1])
        return rec

    def compute_statistics(self, results_list):
        """
        Compute statistics needed for COCO metrics (mAP, AP of individual classes, mAP@IoU_Thresholds, AR)
        Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list) 
                per cateory (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth should be 
                    ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored

        Returns:
            dict: computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
        """
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)
        num_max_detections = len(self.max_detections)

        # -1 for the precision of absent categories
        precision = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))
        recall = -np.ones((num_iou_th, num_classes, num_max_detections))
        scores = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))

        for cls_idx, cls_i in enumerate(self.classes):  # for each class
            for maxDet_idx, maxDet in enumerate(self.max_detections):  # for each maximum number of detections
                results = [r[cls_idx] for r in results_list if cls_idx in r]

                if len(results) == 0:
                    continue

                dt_scores = np.concatenate([r['dtScores'][0:maxDet] for r in results])
                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dt_scores, kind='mergesort')
                dt_scores_sorted = dt_scores[inds]

                # r['dtMatches'] [T, R], where R = sum(all detections)
                dt_matches = np.concatenate([r['dtMatches'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                dt_ignores = np.concatenate([r['dtIgnore'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                self.check_number_of_iou(dt_matches, dt_ignores)
                gt_ignore = np.concatenate([r['gtIgnore'] for r in results])
                num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
                if num_gt == 0:
                    continue

                # ignore cases need to be handled differently for tp and fp
                tps = np.logical_and(dt_matches,  np.logical_not(dt_ignores))
                fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)

                for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each threshold th_ind
                    tp, fp = np.array(tp), np.array(fp)
                    r, p, s = compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                    recall[th_ind, cls_idx, maxDet_idx] = r
                    precision[th_ind, :, cls_idx, maxDet_idx] = p
                    # corresponding score thresholds for recall steps
                    scores[th_ind, :, cls_idx, maxDet_idx] = s

        return {
            'counts': [num_iou_th, num_recall_th, num_classes, num_max_detections],  # [4]
            'recall':   recall,  # [num_iou_th, num_classes, num_max_detections]
            'precision': precision,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
            'scores': scores,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
        }

def compute_stats_single_threshold(
    tp,
    fp,
    dt_scores_sorted,
    recall_thresholds,
    num_gt
):
    """
    Compute recall value, precision curve and scores thresholds
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        tp (np.ndarray): cumsum over true positives [R], R is the number of detections
        fp (np.ndarray): cumsum over false positives [R], R is the number of detections
        dt_scores_sorted (np.ndarray): sorted (descending) scores [R], R is the number of detections
        recall_thresholds (Sequence[float]): recall thresholds which should be evaluated
        num_gt (int): number of ground truth bounding boxes (excluding boxes which are ignored)

    Returns:
        float: overall recall for given IoU value
        np.ndarray: precision values at defined recall values
            [RTH], where RTH is the number of recall thresholds
        np.ndarray: prediction scores corresponding to recall values
            [RTH], where RTH is the number of recall thresholds
    """
    num_recall_th = len(recall_thresholds)

    rc = tp / num_gt
    # np.spacing(1) is the smallest representable epsilon with float
    pr = tp / (fp + tp + np.spacing(1))

    if len(tp):
        recall = rc[-1]
    else:
        # no prediction
        recall = 0

    # array where precision values nearest to given recall th are saved
    precision = np.zeros((num_recall_th,))
    # save scores for corresponding recall value in here
    th_scores = np.zeros((num_recall_th,))
    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist(); precision = precision.tolist()

    # smooth precision curve (create box shape)
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]

    # get indices to nearest given recall threshold (nn interpolation!)
    inds = np.searchsorted(rc, recall_thresholds, side='left')
    try:
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except:
        pass

    return recall, np.array(precision), np.array(th_scores)

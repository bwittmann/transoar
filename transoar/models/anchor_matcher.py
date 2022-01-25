"""
Parts of this code are from torchvision and thus licensed under

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from typing import Sequence, Callable, Tuple
from abc import ABC

import torch
from torch import Tensor
from torch.cuda.amp import autocast

INF = 100  # not really inv but here it is sufficient


class Matcher(ABC):
    BELOW_LOW_THRESHOLD: int = -1
    BETWEEN_THRESHOLDS: int = -2

    def __init__(self, similarity_fn: Callable[[Tensor, Tensor], Tensor]):
        """
        Matches boxes and anchors to each other

        Args:
            similarity_fn: function for similarity computation between
                boxes and anchors
        """
        self.similarity_fn = similarity_fn

    def __call__(self,
                 boxes: torch.Tensor,
                 anchors: torch.Tensor,
                 num_anchors_per_level: Sequence[int],
                 num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches for a single image

        Args:
            boxes: anchors are matches to these boxes (e.g. ground truth)
                [N, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors: anchors to match [M, dims * 2](x1, y1, x2, y2, (z1, z2))
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        if boxes.numel() == 0:
            # no ground truth
            num_anchors = anchors.shape[0]
            match_quality_matrix = torch.tensor([]).to(anchors)
            matches = torch.empty(num_anchors, dtype=torch.int64).fill_(self.BELOW_LOW_THRESHOLD)
            return match_quality_matrix, matches
        else:
            # at least one ground truth
            return self.compute_matches(
                boxes=boxes, anchors=anchors,
                num_anchors_per_level=num_anchors_per_level,
                num_anchors_per_loc=num_anchors_per_loc,
                )

    def compute_matches(self,
                        boxes: torch.Tensor,
                        anchors: torch.Tensor,
                        num_anchors_per_level: Sequence[int],
                        num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches

        Args:
            boxes: anchors are matches to these boxes (e.g. ground truth)
                [N, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors: anchors to match [M, dims * 2](x1, y1, x2, y2, (z1, z2))
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        raise NotImplementedError


class ATSSMatcher(Matcher):
    def __init__(self,
                 num_candidates: int,
                 similarity_fn: Callable[[Tensor, Tensor], Tensor],
                 center_in_gt: bool = True,
                 ):
        """
        Compute matching based on ATSS
        https://arxiv.org/abs/1912.02424
        `Bridging the Gap Between Anchor-based and Anchor-free Detection
        via Adaptive Training Sample Selection`

        Args:
            num_candidates: number of positions to select candidates from
            similarity_fn: function for similarity computation between
                boxes and anchors
            center_in_gt: If diabled, matched anchor center points do not need
                to lie withing the ground truth box.
        """
        super().__init__(similarity_fn=similarity_fn)
        self.num_candidates = num_candidates
        self.min_dist = 0.01
        self.center_in_gt = center_in_gt

    def compute_matches(self,
                        boxes: torch.Tensor,
                        anchors: torch.Tensor,
                        num_anchors_per_level: Sequence[int],
                        num_anchors_per_loc: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to ATTS for a single image
        Adapted from
        (https://github.com/sfzhang15/ATSS/blob/79dfb28bd1/atss_core/modeling/rpn/atss
        /loss.py#L180-L184)

        Args:
            boxes: anchors are matches to these boxes (e.g. ground truth)
                [N, dims * 2](x1, y1, x2, y2, (z1, z2))
            anchors: anchors to match [M, dims * 2](x1, y1, x2, y2, (z1, z2))
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        num_gt = boxes.shape[0]
        num_anchors = anchors.shape[0]

        distances, boxes_center, anchors_center = box_center_dist(boxes, anchors)  # num_boxes x anchors

        # select candidates based on center distance
        candidate_idx = []
        start_idx = 0
        for level, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl

            topk = min(self.num_candidates * num_anchors_per_loc, apl)
            _, idx = distances[:, start_idx: end_idx].topk(topk, dim=1, largest=False)
            # idx shape [num_boxes x topk]
            candidate_idx.append(idx + start_idx)

            start_idx = end_idx

        # [num_boxes x num_candidates] (index of candidate anchors)
        candidate_idx = torch.cat(candidate_idx, dim=1)

        match_quality_matrix = self.similarity_fn(boxes, anchors)  # [num_boxes x anchors]
        candidate_ious = match_quality_matrix.gather(1, candidate_idx)  # [num_boxes, n_candidates]

        # compute adaptive iou threshold
        iou_mean_per_gt = candidate_ious.mean(dim=1)  # [num_boxes]
        iou_std_per_gt = candidate_ious.std(dim=1)  # [num_boxes]
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt  # [num_boxes]
        is_pos = candidate_ious >= iou_thresh_per_gt[:, None]  # [num_boxes x n_candidates]

        if self.center_in_gt:  # can discard all candidates in case of very small objects :/
            # center point of selected anchors needs to lie within the ground truth
            boxes_idx = torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None]\
                .expand_as(candidate_idx).contiguous()  # [num_boxes x n_candidates]
            is_in_gt = center_in_boxes(
                anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist)
            is_pos = is_pos & is_in_gt.view_as(is_pos)  # [num_boxes x n_candidates]

        # in case on anchor is assigned to multiple boxes, use box with highest IoU
        # TODO: think about a better way to do this
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        ious_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        ious_inf[index] = match_quality_matrix.view(-1)[index]
        ious_inf = ious_inf.view_as(match_quality_matrix)

        matched_vals, matches = ious_inf.max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        # print(f"Num matches {(matches >= 0).sum()}, Adapt IoU {iou_thresh_per_gt}")
        return match_quality_matrix, matches


def box_center_dist(boxes1: Tensor, boxes2: Tensor, euclidean: bool = True) -> \
        Tuple[Tensor, Tensor, Tensor]:
    """
    Distance of center points between two sets of boxes

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        euclidean: computed the euclidean distance otherwise it uses the l1
            distance

    Returns:
        Tensor: the NxM matrix containing the pairwise
            distances for every element in boxes1 and boxes2; [N, M]
        Tensor: center points of boxes1
        Tensor: center points of boxes2
    """
    center1 = box_center(boxes1)  # [N, dims]
    center2 = box_center(boxes2)  # [M, dims]

    if euclidean:
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        # before sum: [N, M, dims]
        dists = (center1[:, None] - center2[None]).sum(-1)
    return dists, center1, center2

def box_center(boxes: Tensor) -> Tensor:
    """
    Compute center point of boxes

    Args:
        boxes: bounding boxes (x1, y1, x2, y2, (z1, z2)) [N, dims * 2]

    Returns:
        Tensor: center points [N, dims]
    """
    centers = [(boxes[:, 2] + boxes[:, 0]) / 2., (boxes[:, 3] + boxes[:, 1]) / 2.]
    if boxes.shape[1] == 6:
        centers.append((boxes[:, 5] + boxes[:, 4]) / 2.)
    return torch.stack(centers, dim=1)


def center_in_boxes(center: Tensor, boxes: Tensor, eps: float = 0.01) -> Tensor:
    """
    Checks which center points are within boxes

    Args:
        center: center points [N, dims]
        boxes: boxes [N, dims * 2]
        eps: minimum distance to boarder of boxes

    Returns:
        Tensor: boolean array indicating which center points are within
            the boxes [N]
    """
    axes = []
    axes.append(center[:, 0] - boxes[:, 0])
    axes.append(center[:, 1] - boxes[:, 1])
    axes.append(boxes[:, 2] - center[:, 0])
    axes.append(boxes[:, 3] - center[:, 1])
    if center.shape[1] == 3:
        axes.append(center[:, 2] - boxes[:, 4])
        axes.append(boxes[:, 5] - center[:, 2])
    return torch.stack(axes, dim=1).min(dim=1)[0] > eps


@autocast(enabled=False)
def box_iou(boxes1: Tensor, boxes2: Tensor,  eps: float = 0) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    (Works for Tensors and Numpy Arrays)

    Arguments:
        boxes1: boxes; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        boxes2: boxes; (x1, y1, x2, y2, (z1, z2))[M, dim * 2]
        eps: optional small constant for numerical stability

    Returns:
        iou (Tensor): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2; [N, M]

    See Also:
        :func:`box_iou_3d`, :func:`torchvision.ops.boxes.box_iou`

    Notes:
        Need to compute IoU in float32 (autocast=False) because the
        volume/area can be to large
    """
    # TODO: think about adding additional assert statements to check coordinates x1 <= x2, y1 <= y2, z1 <= z2
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.tensor([]).to(boxes1)
    return box_iou_union_3d(boxes1.float(), boxes2.float(), eps=eps)[0]

def box_iou_union_3d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tuple[Tensor, Tensor]:
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

def box_area_3d(boxes: Tensor) -> Tensor:
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
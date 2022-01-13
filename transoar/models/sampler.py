import torch

from abc import ABC
from typing import List
from torch import Tensor


class HardNegativeSamplerMixin(ABC):
    def __init__(self, pool_size: float = 10):
        self.pool_size = pool_size

    def select_negatives(self, negative: Tensor, num_neg: int,
                         img_labels: Tensor, img_fg_probs: Tensor):
        pool = int(num_neg * self.pool_size)
        pool = min(negative.numel(), pool) # protect against not enough negatives

        # select pool of highest scoring false positives
        _, negative_idx_pool = img_fg_probs[negative].topk(pool, sorted=True)
        negative = negative[negative_idx_pool]

        # select negatives from pool
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        neg_idx_per_image = negative[perm2]

        neg_idx_per_image_mask = torch.zeros_like(img_labels, dtype=torch.uint8)
        neg_idx_per_image_mask[neg_idx_per_image] = 1
        return neg_idx_per_image_mask

class HardNegativeSampler(HardNegativeSamplerMixin):
    def __init__(self, batch_size_per_image: int, positive_fraction: float,
                 min_neg: int = 0, pool_size: float = 10):

        super().__init__(pool_size=pool_size)
        self.min_neg = min_neg
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in target_labels]
        fg_probs = fg_probs.split(anchors_per_image, 0)

        pos_idx = []
        neg_idx = []
        for img_labels, img_fg_probs in zip(target_labels, fg_probs):
            positive = torch.where(img_labels >= 1)[0]
            negative = torch.where(img_labels == 0)[0]

            num_pos = self.get_num_pos(positive)
            pos_idx_per_image_mask = self.select_positives(
                positive, num_pos, img_labels, img_fg_probs)
            pos_idx.append(pos_idx_per_image_mask)

            num_neg = self.get_num_neg(negative, num_pos)
            neg_idx_per_image_mask = self.select_negatives(
                negative, num_neg, img_labels, img_fg_probs)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

    def get_num_pos(self, positive: torch.Tensor) -> int:
        # positive anchor sampling
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        return num_pos

    def get_num_neg(self, negative: torch.Tensor, num_pos: int) -> int:
        # always assume at least one pos anchor was sampled
        num_neg = int(max(1, num_pos) * abs(1 - 1. / float(self.positive_fraction)))
        # protect against not enough negative examples and sample at least one neg if possible
        num_neg = min(negative.numel(), max(num_neg, self.min_neg))
        return num_neg

    def select_positives(self, positive: Tensor, num_pos: int,
                         img_labels: Tensor, img_fg_probs: Tensor):

        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx_per_image = positive[perm1]
        pos_idx_per_image_mask = torch.zeros_like(img_labels, dtype=torch.uint8)
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        return pos_idx_per_image_mask


class HardNegativeSamplerBatched(HardNegativeSampler):
    def __init__(self, batch_size_per_image: int, positive_fraction: float,
                min_neg: int = 0, pool_size: float = 10):
        super().__init__(min_neg=min_neg, batch_size_per_image=batch_size_per_image,
                         positive_fraction=positive_fraction, pool_size=pool_size)
        self._batch_size_per_image = batch_size_per_image

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        batch_size = len(target_labels)
        self.batch_size_per_image = self._batch_size_per_image * batch_size

        target_labels_batch = torch.cat(target_labels, dim=0)

        positive = torch.where(target_labels_batch >= 1)[0]
        negative = torch.where(target_labels_batch == 0)[0]

        num_pos = self.get_num_pos(positive)
        pos_idx = self.select_positives(
            positive, num_pos, target_labels_batch, fg_probs)

        num_neg = self.get_num_neg(negative, num_pos)
        neg_idx = self.select_negatives(
            negative, num_neg, target_labels_batch, fg_probs)

        # Comb Head with sampling concatenates masks after sampling so do not split them here
        # anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in target_labels]
        # return pos_idx.split(anchors_per_image, 0), neg_idx.split(anchors_per_image, 0)
        return [pos_idx], [neg_idx]

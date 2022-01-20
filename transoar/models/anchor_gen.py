import torch
import torch.nn as nn
from typing import Sequence, List, Tuple, Union
from itertools import product


class AnchorGenerator2D(nn.Module):
    def __init__(self, sizes: Sequence[Union[int, Sequence[int]]] = (128, 256, 512),
                 aspect_ratios: Sequence[Union[float, Sequence[float]]] = (0.5, 1.0, 2.0),
                 **kwargs):
        """
        Generator for anchors
        Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/rpn.py

        Args:
            sizes (Sequence[Union[int, Sequence[int]]]): anchor sizes for each feature map
                (length should match the number of feature maps)
            aspect_ratios (Sequence[Union[float, Sequence[float]]]): anchor aspect ratios:
                height/width, e.g. (0.5, 1, 2). if Seq[Seq] is provided, it should have
                the same length as sizes
        """
        super().__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

        self.num_anchors_per_level: List[int] = None

    def cached_grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[int]]) -> List[torch.Tensor]:
        """
        Check if combination was already generated before and return that if possible

        Args:
            grid_sizes (Sequence[Sequence[int]]): spatial sizes of feature maps
            strides (Sequence[Sequence[int]]): stride of each feature map

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
        """
        key = str(grid_sizes + strides)
        if key not in self._cache:
            self._cache[key] = self.grid_anchors(grid_sizes, strides)

        self.num_anchors_per_level = self._cache[key][1]
        return self._cache[key][0]


    def forward(self, image_list: torch.Tensor, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate anchors for given feature maps
        # TODO: update docstring and type
        Args:
            image_list (torch.Tensor): data structure which contains images and their original shapes
            feature_maps (Sequence[torch.Tensor]): feature maps for which anchors need to be generated

        Returns:
            List[Tensor]: list of anchors (for each image inside the batch)
        """
        device = image_list.device
        grid_sizes = list([feature_map.shape[2:] for feature_map in feature_maps])
        image_size = image_list.shape[2:]
        strides = [list((int(i / s) for i, s in zip(image_size, fm_size))) for fm_size in grid_sizes]

        self.set_cell_anchors(dtype=feature_maps[0].dtype, device=feature_maps[0].device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = []
        images_shapes = [img.shape for img in image_list.split(1)]
        for i, x in enumerate(images_shapes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image).to(device) for anchors_per_image in anchors]

        # TODO: check with torchvision if this makes sense (if enabled, anchors are newly generated for each run)
        # # Clear the cache in case that memory leaks.
        # self._cache.clear()
        return anchors

    def get_num_anchors_per_level(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        if self.num_anchors_per_level is None:
            raise RuntimeError("Need to forward features maps before "
                               "get_num_acnhors_per_level can be called")
        return self.num_anchors_per_level

class AnchorGenerator3D(AnchorGenerator2D):
    def __init__(self,
                 sizes: Sequence[Union[int, Sequence[int]]] = (128, 256, 512),
                 aspect_ratios: Sequence[Union[float, Sequence[float]]] = (0.5, 1.0, 2.0),
                 zsizes: Sequence[Union[int, Sequence[int]]] = (4, 4, 4),
                 **kwargs):
        """
        Helper to generate anchors for different input sizes

        Args:
            sizes (Sequence[Union[int, Sequence[int]]]): anchor sizes for each feature map
                (length should match the number of feature maps)
            aspect_ratios (Sequence[Union[float, Sequence[float]]]): anchor aspect ratios:
                height/width, e.g. (0.5, 1, 2). if Seq[Seq] is provided, it should have
                the same length as sizes
            zsizes (Sequence[Union[int, Sequence[int]]]): sizes along z dimension
        """
        super().__init__(sizes, aspect_ratios)
        if not isinstance(zsizes[0], (Sequence, list, tuple)):
            zsizes = (zsizes,) * len(sizes)
        self.zsizes = zsizes

    def grid_anchors(self, grid_sizes: Sequence[Sequence[int]],
                     strides: Sequence[Sequence[int]]) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Distribute anchors over feature maps

        Args:
            grid_sizes (Sequence[Sequence[int]]): spatial sizes of feature maps
            strides (Sequence[Sequence[int]]): stride of each feature map

        Returns:
            List[torch.Tensor]: Anchors for each feature maps
            List[int]: number of anchors per level
        """
        assert len(grid_sizes) == len(strides)
        assert len(grid_sizes) == len(self.cell_anchors)
        anchors = []
        _i = 0
        anchor_per_level = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            size0, size1, size2 = size
            stride0, stride1, stride2 = stride
            dtype, device = base_anchors.dtype, base_anchors.device

            shifts_x = torch.arange(0, size0, dtype=dtype, device=device) * stride0
            shifts_y = torch.arange(0, size1, dtype=dtype, device=device) * stride1
            shifts_z = torch.arange(0, size2, dtype=dtype, device=device) * stride2

            shift_x, shift_y, shift_z = torch.meshgrid(shifts_x, shifts_y, shifts_z)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shift_z = shift_z.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y, shift_z, shift_z), dim=1)

            _anchors = (shifts.view(-1, 1, 6) + base_anchors.view(1, -1, 6)).reshape(-1, 6)
            anchors.append(_anchors)
            anchor_per_level.append(_anchors.shape[0])

            _i += 1
        return anchors, anchor_per_level


class AnchorGenerator3DS(AnchorGenerator3D):
    def __init__(self,
                 width: Sequence[Union[int, Sequence[int]]],
                 height: Sequence[Union[int, Sequence[int]]],
                 depth: Sequence[Union[int, Sequence[int]]],
                 **kwargs,
                 ):
        """
        Helper to generate anchors for different input sizes
        Uses a different parametrization of anchors
        (if Sequence[int] is provided it is interpreted as one 
        value per feature map size)

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension
        """
        # TODO: check width and height statements
        super().__init__()
        if not isinstance(width[0], Sequence):
            width = [(w,) for w in width]
        if not isinstance(height[0], Sequence):
            height = [(h,) for h in height]
        if not isinstance(depth[0], Sequence):
            depth = [(d,) for d in depth]
        self.width = width
        self.height = height
        self.depth = depth
        assert len(self.width) == len(self.height) == len(self.depth)

    def set_cell_anchors(self, dtype: torch.dtype, device: Union[torch.device, str] = "cpu") -> None:
        """
        Compute anchors for all pairs of scales and ratios and save them inside :param:`cell_anchors`
        if they were not computed before

        Args:
            dtype (torch.dtype): data type of anchors
            device (Union[torch.device, str]): target device of anchors

        Returns:
            None (result is saved into :param:`self.cell_anchors`)
        """
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(w, h, d, dtype, device)
            for w, h, d in zip(self.width, self.height, self.depth)
        ]
        self.cell_anchors = cell_anchors

    @staticmethod
    def generate_anchors(width: Tuple[int],
                         height: Tuple[int],
                         depth: Tuple[int],
                         dtype: torch.dtype = torch.float,
                         device: Union[torch.device, str] = "cpu") -> torch.Tensor:
        """
        Generate anchors for given width, height and depth sizes

        Args:
            width: sizes along width dimension
            height: sizes along height dimension
            depth: sizes along depth dimension

        Returns:
            Tensor: anchors of shape [n(width) * n(height) * n(depth) , dim * 2]
        """
        all_sizes = torch.tensor(list(product(width, height, depth)),
                                 dtype=dtype, device=device) / 2
        anchors = torch.stack(
            [-all_sizes[:, 0], -all_sizes[:, 1], all_sizes[:, 0], all_sizes[:, 1],
             -all_sizes[:, 2], all_sizes[:, 2]], dim=1
            )
        return anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        Number of anchors per resolution

        Returns:
            List[int]: number of anchors per positions for each resolution
        """
        return [len(w) * len(h) * len(d) 
                for w, h, d in zip(self.width, self.height, self.depth)]

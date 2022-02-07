import torch
import torch.nn as nn

from transoar.models.build import build_pos_enc, build_def_attn_encoder

from transoar.models.head import DetectionHeadHNMNative, ClsHead, RegHead, SegHead
from transoar.models.anchor_gen import AnchorGenerator3DS
from transoar.models.sampler import HardNegativeSamplerBatched
from transoar.models.coder import BoxCoderND
from transoar.models.anchor_matcher import ATSSMatcher, box_iou
from transoar._C import nms


class RetinaUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Build encoder and decoder
        encoder = Encoder(config)
        decoder_channels = encoder.out_channels
        decoder_strides = encoder.out_strides

        decoder = Decoder(config, decoder_channels, decoder_strides)

        # Build def attn encoder
        if config['use_def_attn']:
            def_attn_encoder = build_def_attn_encoder(config)
            pos_enc = build_pos_enc(config)

            # Build input projection to attn encoder
            input_proj_list = []
            for _ in range(config['num_feature_levels']):
                input_proj_list.append(nn.Sequential(
                    nn.Conv3d(config['fpn_channels'], config['hidden_dim'], kernel_size=1),
                    nn.GroupNorm(32, config['hidden_dim']),
                ))

            self.input_proj = nn.ModuleList(input_proj_list)

        # Build detection and segmentation head
        cls_head = ClsHead(config)
        reg_head = RegHead(config)

        sampler = HardNegativeSamplerBatched(
            config['batch_size_per_image'], config['positive_fraction'], config['min_neg'],
            config['pool_size']
        )
        box_coder = BoxCoderND(weights=(1.,) * 6)

        detection_head = DetectionHeadHNMNative(
            classifier=cls_head, regressor=reg_head,
            coder=box_coder, sampler=sampler
        )

        segmentation_head = SegHead(config)

        # Build anchor generator
        anchor_gen = AnchorGenerator3DS(
            config['width'], config['height'], config['depth'], stride=config['stride']
        )

        # Build matcher
        matcher = ATSSMatcher(
            num_candidates=config['num_candidates'], center_in_gt=config['center_in_gt'], similarity_fn=box_iou
        )

        self.decoder_levels = config['decoder_levels']

        self.encoder = encoder
        self.decoder = decoder

        if config['use_def_attn']:
            self.use_def_attn = config['use_def_attn']
            self.def_attn_encoder = def_attn_encoder
            self.pos_enc = pos_enc

        self.head = detection_head
        self.segmenter = segmentation_head

        self.anchor_generator = anchor_gen
        self.proposal_matcher = matcher

        self.num_foreground_classes = config ['classifier_classes']
        self.score_thresh = config['score_thresh']
        self.topk_candidates = config['topk_candidates']
        self.detections_per_img = config['detections_per_img']
        self.remove_small_boxes = config['remove_small_boxes']
        self.nms_thresh = config['nms_thresh']


    def train_step(self, img, targets, evaluation=False):
        target_boxes = targets["target_boxes"]
        target_classes = targets["target_classes"]
        target_seg = targets["target_seg"]

        pred_detection, anchors, pred_seg = self(img)
        labels, matched_gt_boxes = self.assign_targets_to_anchors(
            anchors, target_boxes, target_classes
        )

        losses = {}
        head_losses, pos_idx, neg_idx = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors)
        losses.update(head_losses)

        if self.segmenter is not None:
            losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if evaluation:
            prediction = self.postprocess_for_inference(
                images=img,
                pred_detection=pred_detection,
                pred_seg=pred_seg,
                anchors=anchors,
            )
        else:
            prediction = None

        return losses, prediction
    
    @torch.no_grad()
    def postprocess_for_inference(self, images, pred_detection, pred_seg, anchors):
        image_shapes = [images.shape[2:]] * images.shape[0]
        boxes, probs, labels = self.postprocess_detections(
            pred_detection=pred_detection,
            anchors=anchors,
            image_shapes=image_shapes,
        )
        prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}

        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
        return prediction

    def forward(self, inp):
        features_maps_all = self.decoder(self.encoder(inp))
        feature_maps_red = [features_maps_all[i] for i in self.decoder_levels]

        if self.use_def_attn:
            srcs = []
            masks = []
            pos = []
            for idx, fmap in enumerate(feature_maps_red):
                srcs.append(self.input_proj[idx](fmap))
                mask = torch.zeros_like(fmap[:, 0], dtype=torch.bool) # False is not masked
                masks.append(mask)
                pos.append(self.pos_enc(mask))
            
            feature_maps_head = self.def_attn_encoder(srcs, masks, pos)
        else:
            feature_maps_head = feature_maps_red

        pred_detection = self.head(feature_maps_head)
        anchors = self.anchor_generator(inp, feature_maps_head)

        pred_seg = self.segmenter(features_maps_all) if self.segmenter is not None else None
        return pred_detection, anchors, pred_seg

    @torch.no_grad()
    def assign_targets_to_anchors(self, anchors, target_boxes, target_classes):

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, gt_boxes, gt_classes in zip(anchors, target_boxes, target_classes):
            # indices of ground truth box for each proposal
            match_quality_matrix, matched_idxs = self.proposal_matcher(
                gt_boxes,
                anchors_per_image,
                num_anchors_per_level=self.anchor_generator.get_num_anchors_per_level(),
                num_anchors_per_loc=self.anchor_generator.num_anchors_per_location()[0]
            )

            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            if match_quality_matrix.numel() > 0:
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # Positive (negative indices can be ignored because they are overwritten in the next step)
                # this influences how background class is handled in the input!!!! (here +1 for background)
                labels_per_image = gt_classes[matched_idxs.clamp(min=0)].to(dtype=anchors_per_image.dtype)
                labels_per_image = labels_per_image + 1
            else:
                num_anchors_per_image = anchors_per_image.shape[0]
                # no ground truth => no matches, all background
                matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
                labels_per_image = torch.zeros(num_anchors_per_image).to(anchors_per_image)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds - only for IoU matcher
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def postprocess_detections(self, pred_detection, anchors, image_shapes):
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in anchors]
        pred_detection = self.head.postprocess_for_inference(pred_detection, anchors)
        pred_boxes, pred_probs = pred_detection["pred_boxes"], pred_detection["pred_probs"]

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        all_boxes, all_probs, all_labels = [], [], []
        # iterate over images
        for boxes, probs, image_shape in zip(pred_boxes, pred_probs, image_shapes):
            boxes, probs, labels = self.postprocess_detections_single_image(boxes, probs, image_shape)
            all_boxes.append(boxes)
            all_probs.append(probs)
            all_labels.append(labels)
        return all_boxes, all_probs, all_labels

    def postprocess_detections_single_image(self, boxes, probs, image_shape):
        assert boxes.shape[0] == probs.shape[0]
        boxes = clip_boxes_to_image_(boxes, image_shape)
        probs = probs.flatten()

        if self.topk_candidates is not None:
            num_topk = min(self.topk_candidates, boxes.size(0))
            probs, idx = probs.sort(descending=True)
            probs, idx = probs[:num_topk], idx[:num_topk]
        else:
            idx = torch.arange(probs.numel())

        if self.score_thresh is not None:
            keep_idxs = probs > self.score_thresh
            probs, idx = probs[keep_idxs], idx[keep_idxs]

        anchor_idxs = idx // self.num_foreground_classes
        labels = idx % self.num_foreground_classes
        boxes = boxes[anchor_idxs]

        if self.remove_small_boxes is not None:
            keep = remove_small_boxes(boxes, min_size=self.remove_small_boxes)
            boxes, probs, labels = boxes[keep], probs[keep], labels[keep]

        keep = batched_nms(boxes, probs, labels, self.nms_thresh)
        
        if self.detections_per_img is not None:
            keep = keep[:self.detections_per_img]

        return boxes[keep], probs[keep], labels[keep]

    @torch.no_grad()
    def inference_step(self, images):
        pred_detection, anchors, pred_seg = self(images)
        prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            anchors=anchors,
        )
        return prediction


def clip_boxes_to_image_(boxes, img_shape):
    s0, s1, s2 = img_shape
    boxes[..., 0::6].clamp_(min=0, max=s0)
    boxes[..., 1::6].clamp_(min=0, max=s1)
    boxes[..., 2::6].clamp_(min=0, max=s0)
    boxes[..., 3::6].clamp_(min=0, max=s1)
    boxes[..., 4::6].clamp_(min=0, max=s2)
    boxes[..., 5::6].clamp_(min=0, max=s2)
    return boxes

def remove_small_boxes(boxes, min_size) :
    if boxes.shape[1] == 4:
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
    else:
        ws, hs, ds = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 5] - boxes[:, 4]
        keep = (ws >= min_size) & (hs >= min_size) & (ds >= min_size)
    keep = torch.where(keep)[0]
    return keep

def batched_nms(boxes, scores, idxs, iou_threshold: float):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)

# def nms(boxes, scores, thresh):
#     ious = box_iou(boxes, boxes)
#     _, _idx = torch.sort(scores, descending=True)
    
#     keep = []
#     while _idx.nelement() > 0:
#         keep.append(_idx[0])
#         # get all elements that were not matched and discard all others.
#         non_matches = torch.where((ious[_idx[0]][_idx] <= thresh))[0]
#         _idx = _idx[non_matches]
#     return torch.tensor(keep).to(boxes).long()


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self._config = config
        in_channels = config['in_channels']
        out_channels = config['start_channels']

        num_stages = len(config['conv_kernels'])
        self._out_stages = list(range(num_stages))

        self.out_channels = []
        self.out_strides = []
        self._stages = nn.ModuleList()
        for stage_id in range(num_stages):
            self.out_channels.append(out_channels)

            if len(self.out_strides) == 0:
                self.out_strides.append(config['strides'][stage_id])
            else:
                self.out_strides.append(config['strides'][stage_id]) # * self.out_strides[-1])

            
            stage = EncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config['conv_kernels'][stage_id],
                stride=config['strides'][stage_id]
            )
            self._stages.append(stage)

            in_channels = out_channels
            out_channels = out_channels * 2

            if out_channels > config['max_channels']:
                out_channels = config['max_channels']

    def forward(self, x):
        outputs = []
        for stage_id, module in enumerate(self._stages):
            x = module(x)
            if stage_id in self._out_stages:
                outputs.append(x)
        return outputs 

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=1,
        bias=False,
        affine=True,
        eps=1e-05

    ):
        super().__init__()

        conv_block_1 = [
            nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        conv_block_2 = [
            nn.Conv3d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                bias=bias
            ),
            nn.InstanceNorm3d(num_features=out_channels, affine=affine, eps=eps),
            nn.ReLU(inplace=True)
        ]

        self._block = nn.Sequential(
            *conv_block_1,
            *conv_block_2
        )

    def forward(self, x):
        return self._block(x)

class Decoder(nn.Module):
    def __init__(self, config, encoder_out_channels, strides):
        super().__init__() 
        self._num_levels = len(encoder_out_channels)

        decoder_out_channels = torch.clip(torch.tensor(encoder_out_channels), max=(config['fpn_channels'])).tolist()

        # Lateral 
        self._lateral = nn.ModuleList()
        for in_channels, out_channels in zip(encoder_out_channels, decoder_out_channels):
            self._lateral.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

        # Out
        self._out = nn.ModuleList()
        for out_channels in decoder_out_channels:
            self._out.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))

        #  Up
        self._up = nn.ModuleList()
        for level in range(1, len(decoder_out_channels)):
            self._up.append(nn.ConvTranspose3d(
                in_channels=decoder_out_channels[level], out_channels=decoder_out_channels[level-1],
                kernel_size=config['strides'][level], stride=config['strides'][level]
                ))

    def forward(self, x):
        out_list = []

        # Forward lateral
        fpn_maps = [self._lateral[level](fm) for level, fm in enumerate(x)]

        # Forward up
        for idx, x in enumerate(reversed(fpn_maps), 1):
            level = self._num_levels - idx - 1

            if idx != 1:
                x = x + up

            if idx != self._num_levels:
                up = self._up[level](x)

            out_list.append(x)

        # Forward out
        out_list = [self._out[level](fm) for level, fm in enumerate(reversed(out_list))]

        return out_list

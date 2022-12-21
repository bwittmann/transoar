import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph


from transoar.models.retinanet.head import DetectionHeadHNMNative, ClsHead, RegHead, SegHead
from transoar.models.retinanet.gnn import GCN
from transoar.models.retinanet.loss import GIoULoss
from transoar.models.backbones.attn_fpn import AttnFPN
from transoar.models.anchors.anchor_gen import AnchorGenerator3DS
from transoar.models.sampler import HardNegativeSamplerBatched
from transoar.models.coder import BoxCoderND
from transoar.models.anchors.anchor_matcher import ATSSMatcher, box_iou
from transoar._C import nms


class RetinaUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get configs for backbone and neck
        config_backbone = config['backbone']
        config_neck = config['neck']

        # Build AttnFPN backbone
        self.attn_fpn = AttnFPN(config_backbone)

        # Build GNN
        self.gnn = GCN(
            in_dim=config['neck']['head_channels'],
            h_dim=config['neck']['head_channels'],
            out_dim=6,
            num_gnn_layers=2,
            p_drop=0.0
        )
        self.gnn_loss = GIoULoss()
        self.use_graph = config_neck['use_graph']

        # Build detection and segmentation head
        cls_head = ClsHead(config_neck)
        reg_head = RegHead(config_neck)

        sampler = HardNegativeSamplerBatched(
            config_neck['batch_size_per_image'], config_neck['positive_fraction'],
            config_neck['min_neg'], config_neck['pool_size']
        )
        box_coder = BoxCoderND(weights=(1.,) * 6)

        detection_head = DetectionHeadHNMNative(
            classifier=cls_head, regressor=reg_head,
            coder=box_coder, sampler=sampler
        )

        segmentation_head = SegHead(config_neck)

        # Build anchor generator
        anchor_gen = AnchorGenerator3DS(
            config_neck['width'], config_neck['height'], config_neck['depth'], stride=config_neck['stride']
        )

        # Build matcher
        matcher = ATSSMatcher(
            num_candidates=config_neck['num_candidates'], center_in_gt=config_neck['center_in_gt'], similarity_fn=box_iou
        )

        self.input_levels = config_neck['input_levels']

        self.head = detection_head
        self.segmenter = segmentation_head if config['backbone']['use_seg_proxy_loss'] else None

        self.anchor_generator = anchor_gen
        self.proposal_matcher = matcher

        self.num_foreground_classes = config_neck['classifier_classes']
        self.score_thresh = config_neck['score_thresh']
        self.topk_candidates = config_neck['topk_candidates']
        self.detections_per_img = config_neck['detections_per_img']
        self.remove_small_boxes = config_neck['remove_small_boxes']
        self.nms_thresh = config_neck['nms_thresh']

    def train_step(self, img, targets, evaluation=False):
        target_boxes = targets["target_boxes"]
        target_classes = targets["target_classes"]
        target_seg = targets["target_seg"]

        pred_detection, anchors, pred_seg = self(img)
        labels, matched_gt_boxes = self.assign_targets_to_anchors(
            anchors, target_boxes, target_classes
        )

        # general detection losses
        losses = {}
        head_losses, _, _ = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors)
        losses.update(head_losses)

        # just return best pred with hightest score per class to circumvent non diff nms
        preds_per_batch = int(pred_detection['box_logits'].shape[0] / img.shape[0])

        batch_logits = torch.split(pred_detection['box_logits'], preds_per_batch)
        batch_deltas = torch.split(pred_detection['box_deltas'], preds_per_batch)
        batch_features = torch.split(pred_detection['box_features'], preds_per_batch)

        best_ids = [logits_batch.softmax(dim=1).argmax(dim=0) for logits_batch in batch_logits]
        best_anchors = [anchors[ids] for anchors, ids in zip(anchors, best_ids)]
        best_logits = [logits[ids] for logits, ids in zip(batch_logits, best_ids)]
        best_deltas = [batch_delta[ids] for batch_delta, ids in zip(batch_deltas, best_ids)]
        best_features = [batch_feature[ids] for batch_feature, ids in zip(batch_features, best_ids)]
        
        if self.use_graph:
            # graph construction
            graphs = []
            for boxes, anchors, features in zip(best_deltas, best_anchors, best_features):
                boxes_cpos = self.head.coder.decode_single(boxes, anchors)[:, :3]   # TODO
                edge_index = knn_graph(boxes_cpos, k=10).to(device=features.device)
                graphs.append(Data(x=features.float(), edge_index=edge_index.long()))
            graph_batch = Batch().from_data_list(graphs)

            best_deltas = self.gnn(graph_batch)
            gnn_pred_detection = {
                'box_deltas': torch.cat([deltas[classes] for deltas, classes in zip(torch.split(best_deltas, 20), target_classes)]),
                'box_logits': torch.cat([logits[classes] for logits, classes in zip(best_logits, target_classes)])
            }
            best_anchors = [anchors[classes] for anchors, classes in zip(best_anchors, target_classes)]

            # graph losses
            gnn_loss, _, _ = self.head.compute_loss(
                gnn_pred_detection, target_classes, target_boxes, best_anchors, gnn=True)
            losses.update(gnn_loss)

            pred_detection_final = self.head.postprocess_for_inference(gnn_pred_detection, best_anchors)
            prediction = {
                'pred_boxes': [pred_detection_final['pred_boxes'][:target_classes[0].shape[0]], pred_detection_final['pred_boxes'][target_classes[0].shape[0]:]],
                'pred_scores': [pred_detection_final['pred_probs'].max(dim=1)[0][:target_classes[0].shape[0]], pred_detection_final['pred_probs'].max(dim=1)[0][target_classes[0].shape[0]:]],
                'pred_labels': target_classes # TODO
            }

            # graph loss
            # proc_boxes = torch.cat([preds[classes] for preds, classes in zip(prediction['pred_boxes'], target_classes)])
            # losses['gnn'] = self.gnn_loss(proc_boxes, torch.cat(target_boxes))
        else:
            pred_detection_mod = {
                'box_deltas': torch.cat(best_deltas),
                'box_logits': torch.cat(best_logits)
            }

            pred_detection_final = self.head.postprocess_for_inference(pred_detection_mod, best_anchors)
            prediction = {
                'pred_boxes': torch.split(pred_detection_final['pred_boxes'], 20), # TODO
                'pred_scores': [scores.max(dim=1)[0] for scores in torch.split(pred_detection_final['pred_probs'], 20)],
                'pred_labels': [torch.arange(0, 20) for _ in range(img.shape[0])] # TODO
            }

        # if self.segmenter is not None:
        #     losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        # if evaluation:
        #     prediction = self.postprocess_for_inference(
        #         images=img,
        #         pred_detection=pred_detection,
        #         pred_seg=pred_seg,
        #         anchors=anchors,
        #     )
        # else:
        #     prediction = None

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
        features_maps_all = self.attn_fpn(inp)
        feature_maps_head = [features_maps_all[i] for i in self.input_levels]

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

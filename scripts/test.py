"""Script to evalute performance on the val and test set."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from transoar.utils.io import load_json, write_json
from transoar.utils.bboxes import box_xyxyzz_to_cxcyczwhd
from transoar.utils.visualization import save_pred_visualization
from transoar.data.dataloader import get_loader
from transoar.evaluator import DetectionEvaluator
from transoar.models.retinanet.retina_unet import RetinaUNet

class Tester:

    def __init__(self, args):
        path_to_run = Path('./runs/' + args.run)
        config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._full_labeled = args.full_labeled
        self._class_dict = config['labels']
        self._device = 'cuda' if args.num_gpu >= 0 else 'cpu'

        # Get path to checkpoint
        avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
        avail_checkpoints.sort(key=lambda x: len(str(x)))
        if args.last:
            path_to_ckpt = avail_checkpoints[0]
        else:
            path_to_ckpt = avail_checkpoints[-1]

        # Build necessary components
        self._set_to_eval = 'val' if args.val else 'test'
        self._test_loader = get_loader(config, self._set_to_eval, batch_size=1)

        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values()),
            classes_small=config['labels_small'],
            classes_mid=config['labels_mid'],
            classes_large=config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=False
        )
        self._model = RetinaUNet(config).to(device=self._device)

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        self._path_to_results.mkdir(parents=True, exist_ok=True)

        if self._save_preds:
            self._path_to_vis = self._path_to_results / ('vis_' + self._set_to_eval)
            self._path_to_vis.mkdir(parents=False, exist_ok=True)
  
    def run(self):
        # with torch.no_grad():
        for idx, (data, _, bboxes, seg_mask) in enumerate(tqdm(self._test_loader)):
            # Put data to gpu
            data = data.to(device=self._device)
        
            targets = defaultdict(list)
            for item in bboxes:
                targets['target_boxes'].append(item[0].to(dtype=torch.float, device=self._device))
                targets['target_classes'].append(item[1].to(device=self._device))
            targets['target_seg'] = seg_mask.squeeze(1).to(device=self._device)

            # Only use complete data for performance evaluation
            if self._full_labeled:
                if targets['target_classes'][0].shape[0] < len(self._class_dict):
                    continue

            ### grad cam ###
            import cv2
            import torch.nn.functional as F

            _, prediction, features_maps_all, keep, pred_det = self._model.train_step(data, targets, evaluation=True)
            seg_mask_raw = torch.permute(seg_mask.squeeze(), (1, 0, 2)).short().cpu()

            labels = prediction['pred_labels'][0].cpu()
            scores = prediction['pred_scores'][0].cpu()
            best_ids = torch.tensor(np.unique(labels.numpy(), return_index=True)[1])
            global_best_ids = keep[best_ids]

            for best_id in zip(global_best_ids):
                class_ = pred_det['box_logits'][best_id].argmax().cpu()
                pred_det['box_logits'][best_id][class_].backward(retain_graph=True)

                gradients_list = list(zip(['P5', 'P4', 'P3', 'P2'], self._model.attn_fpn._decoder.get_activations_gradient()))
                self._model.attn_fpn._decoder.gradients = []
                active_layer = torch.tensor([grad[1].sum() for grad in gradients_list]).abs().argmax().item()

                fmap_id, gradients = gradients_list[active_layer]
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4]).cpu()
                assert gradients.sum() != 0
                #print(class_, fmap_id, pooled_gradients.sum(), best_id)

                activations = features_maps_all[fmap_id].detach().cpu()
                assert list(activations.shape[-3:]) == list(gradients.shape[-3:])

                for i in range(384):
                    activations[:, i, :, :] *= pooled_gradients[i]

                heatmap = torch.mean(activations, dim=1).squeeze().relu()
                heatmap /= torch.max(heatmap)

                heatmap = F.interpolate(heatmap[None, None], [160, 160, 256]).squeeze()
                heatmap = torch.permute(heatmap, (1, 0, 2)) * 255

                seg_mask = seg_mask_raw.clone()
                seg_mask[seg_mask == class_ + 1] = -1
                seg_mask[seg_mask > 0] = 50
                seg_mask[seg_mask == -1] = 240

                for idx_, (seg_frame, heat_frame) in enumerate(zip(seg_mask, heatmap)):
                    seg_frame_rgb =  seg_frame.unsqueeze(-1).repeat(1, 1, 3)

                    if True:
                        heat_frame = heatmap.mean(dim=0)
                        min_val, max_val = heat_frame.min(), heat_frame.max()
                        heat_frame = ((heat_frame - min_val) / max_val) * 255

                    # if idx_ % 5 != 0:
                    if idx_ not in [119, 96]:
                        continue

                    b_channel, g_channel, r_channel = cv2.split(torch.zeros_like(heat_frame).unsqueeze(-1).repeat(1, 1, 3).numpy())
                    r_channel.fill(255)
                    alpha_channel = heat_frame.short().numpy()
                    img_BGRA = cv2.merge((b_channel.astype(np.int16), g_channel.astype(np.int16), r_channel.astype(np.int16), alpha_channel))

                    path = Path('/home/home/supro_bastian/download')
                    cv2.imwrite(str(path / f'{idx}_{class_}_frame{idx_}_seg.png'), seg_frame_rgb.numpy())
                    cv2.imwrite(str(path / f'{idx}_{class_}_frame{idx_}_attn.png'), img_BGRA)

            continue

            activations = activations['P5']

            for i in range(384):
                activations[:, i, :, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze().relu().cpu()
            heatmap /= torch.max(heatmap)
            heatmap = F.interpolate(heatmap[None, None], [160, 160, 256]).squeeze()
            heatmap = torch.permute(heatmap, (1, 0, 2))

            heatmap = heatmap.mean(dim=0)
            min_val, max_val = heatmap.min(), heatmap.max()
            heatmap = ((heatmap - min_val) / max_val) * 255
            
            seg_mask_raw = torch.permute(seg_mask.squeeze(), (1, 0, 2)).short()
            seg_mask = seg_mask_raw.clone()
            seg_mask[seg_mask == class_] = -1
            seg_mask[seg_mask > 0] = 50
            seg_mask[seg_mask == -1] = 240


            for idx_, seg_frame in enumerate(seg_mask):
                seg_frame_rgb =  seg_frame.unsqueeze(-1).repeat(1, 1, 3)

                if idx_ % 5 != 0:
                    continue

                b_channel, g_channel, r_channel = cv2.split(torch.zeros_like(heatmap).unsqueeze(-1).repeat(1, 1, 3).numpy())
                r_channel.fill(255)
                alpha_channel = heatmap.short().numpy()
                img_BGRA = cv2.merge((b_channel.astype(np.int16), g_channel.astype(np.int16), r_channel.astype(np.int16), alpha_channel))

                # path = Path('/home/home/supro_bastian/download')
                # cv2.imwrite(str(path / f'{idx}_frame{idx_}_seg.png'), seg_frame_rgb.numpy())
                # cv2.imwrite(str(path / f'{idx}_frame{idx_}_attn.png'), img_BGRA)
           

            continue
            pred_boxes = predictions['pred_boxes'][0].detach().cpu().numpy()
            pred_classes = predictions['pred_labels'][0].detach().cpu().numpy()
            pred_scores = predictions['pred_scores'][0].detach().cpu().numpy()
            gt_boxes = targets['target_boxes'][0].detach().cpu().numpy()
            gt_classes = targets['target_classes'][0].detach().cpu().numpy()

            # Evaluate validation predictions based on metric
            self._evaluator.add(
                pred_boxes=[pred_boxes],
                pred_classes=[pred_classes],
                pred_scores=[pred_scores],
                gt_boxes=[gt_boxes],
                gt_classes=[gt_classes],
            )

            # Just take most confident prediction per class
            best_ids = []
            for class_ in np.unique(pred_classes):
                class_ids = (pred_classes == class_).nonzero()[0]
                max_scoring_idx = pred_scores[class_ids].argmax()
                best_ids.append(class_ids[max_scoring_idx])
            best_ids = torch.tensor(best_ids)

            if self._save_preds:
                save_pred_visualization(
                    box_xyxyzz_to_cxcyczwhd(pred_boxes[best_ids], data.shape[2:]), pred_classes[best_ids] + 1, 
                        box_xyxyzz_to_cxcyczwhd(pred_boxes[best_ids], data.shape[2:]), pred_classes[best_ids] + 1, 
                    box_xyxyzz_to_cxcyczwhd(pred_boxes[best_ids], data.shape[2:]), pred_classes[best_ids] + 1, 
                    box_xyxyzz_to_cxcyczwhd(gt_boxes, data.shape[2:]), gt_classes + 1, 
                        box_xyxyzz_to_cxcyczwhd(gt_boxes, data.shape[2:]), gt_classes + 1, 
                    box_xyxyzz_to_cxcyczwhd(gt_boxes, data.shape[2:]), gt_classes + 1, 
                    seg_mask[0], self._path_to_vis, self._class_dict, idx
                )

            # Get and store final results
            metric_scores = self._evaluator.eval()
            write_json(metric_scores, self._path_to_results / ('results_' + self._set_to_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in transoar/runs.')
    parser.add_argument('--num_gpu', type=int, default=-1, help='Use model_last instead of model_best.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions.')
    parser.add_argument('--full_labeled', action='store_true', help='Use only fully labeled data.')
    parser.add_argument('--coco_map', action='store_true', help='Use coco map.')
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

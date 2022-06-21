"""Script to evalute performance on the val and test set."""

import os
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from transoar.utils.io import load_json, write_json
from transoar.utils.visualization import save_attn_visualization, save_pred_visualization
from transoar.data.dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference

class Tester:

    def __init__(self, args):
        path_to_run = Path('./runs/' + args.run)
        config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._save_attn_map = args.save_attn_map
        self._full_labeled = args.full_labeled
        self._class_dict = config['labels']

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
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

        self._model = TransoarNet(config).to(device=self._device)

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
        if self._save_attn_map:
            backbone_features_list, dec_attn_weights_list = [], []
            
            # Register hooks to efficiently access relevant weights
            hooks = [
                self._model._backbone._decoder._out[0].register_forward_hook(
                    lambda self, input, output: backbone_features_list.append(output)
                ),
                self._model._neck.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_list.append(output[-1])
                )
            ]
    
        with torch.no_grad():
            for idx, (data, mask, bboxes, seg_mask) in enumerate(tqdm(self._test_loader)):
                # Put data to gpu
                data, mask = data.to(device=self._device), mask.to(device=self._device)
            
                targets = {
                    'boxes': bboxes[0][0].to(dtype=torch.float, device=self._device),
                    'labels': bboxes[0][1].to(device=self._device)
                }

                # Only use complete data for performance evaluation
                if targets['labels'].shape[0] < len(self._class_dict):
                    continue

                # Make prediction
                out = self._model(data)

                # Format out to fit evaluator and estimate best predictions per class
                pred_boxes, pred_classes, pred_scores = inference(out, len(self._class_dict))
                gt_boxes = [targets['boxes'].detach().cpu().numpy()]
                gt_classes = [targets['labels'].detach().cpu().numpy()]

                # Add pred to evaluator
                self._evaluator.add(
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes
                )

                if self._save_preds:
                    save_pred_visualization(
                        pred_boxes[0], pred_classes[0], gt_boxes[0], gt_classes[0], seg_mask[0], 
                        self._path_to_vis, self._class_dict, idx
                    )

                if self._save_attn_map:
                    # Get current attn weights
                    backbone_features = backbone_features_list.pop(-1).squeeze()
                    dec_attn_weights = dec_attn_weights_list.pop(-1).squeeze()

                    save_attn_visualization(
                        out, backbone_features, dec_attn_weights, list(data.shape[-3:]),
                        seg_mask[0],
                        idx
                    )

            # Get and store final results
            # [torch.tensor([id_ for score, id_ in query_info[c]]).unique().shape for c in query_info.keys()]
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
    parser.add_argument('--save_attn_map', action='store_true', help='Saves attention maps.')
    parser.add_argument('--full_labeled', action='store_true', help='Use only fully labeled data.')
    parser.add_argument('--coco_map', action='store_true', help='Use coco map.')
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

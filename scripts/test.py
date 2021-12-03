"""Script to evalute performance on the val and test set."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from transoar.utils.io import load_json, write_json
from transoar.utils.visualization import save_pred_visualization
from transoar.data.dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference

class Tester:

    def __init__(self, args):
        path_to_run = Path(args.run)
        config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._class_dict = config['labels']
        self._device = 'cuda:' + str(args.num_gpu)

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

        self._evaluator = DetectionEvaluator(classes=list(config['labels'].values()))
        self._model = TransoarNet(config).to(device=self._device)

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        self._path_to_vis = self._path_to_results / ('vis_' + self._set_to_eval)
        self._path_to_vis.mkdir(parents=True, exist_ok=True)
  
    def run(self):
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
                out = self._model(data, mask)

                # Add pred to evaluator
                pred_boxes, pred_classes, pred_scores = inference(out)
                gt_boxes = [targets['boxes'].detach().cpu().numpy()]
                gt_classes = [targets['labels'].detach().cpu().numpy()]

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

            # Get and store final results
            metric_scores = self._evaluator.eval()
            write_json(metric_scores, self._path_to_results / ('results_' + self._set_to_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Path to experiment in transoar/runs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Use model_last instead of model_best.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--last', action='store_true', help='Use model_last instead of model_best.')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions.')
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

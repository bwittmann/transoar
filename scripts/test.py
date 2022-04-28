"""Script to evalute performance on the val and test set."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from transoar.utils.io import load_json, write_json
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
        self._device = 'cuda:' + str(args.num_gpu) if args.num_gpu >= 0 else 'cpu'

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
            sparse_results=True
        )
        self._model = RetinaUNet(config).to(device=self._device)

        print(sum(p.numel() for p in self._model.parameters() if p.requires_grad))

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
        with torch.no_grad():
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

                # Make prediction
                _, predictions = self._model.train_step(data, targets, evaluation=True)

                # Evaluate validation predictions based on metric
                self._evaluator.add(
                    pred_boxes=[boxes.detach().cpu().numpy() for boxes in predictions['pred_boxes']],
                    pred_classes=[classes.detach().cpu().numpy() for classes in predictions['pred_labels']],
                    pred_scores=[scores.detach().cpu().numpy() for scores in predictions['pred_scores']],
                    gt_boxes=[gt_boxes.detach().cpu().numpy() for gt_boxes in targets['target_boxes']],
                    gt_classes=[gt_classes.detach().cpu().numpy() for gt_classes in targets['target_classes']],
                )

                # if self._save_preds:
                #     save_pred_visualization(
                #         pred_boxes[0], pred_classes[0], gt_boxes[0], gt_classes[0], seg_mask[0], 
                #         self._path_to_vis, self._class_dict, idx
                #     )

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

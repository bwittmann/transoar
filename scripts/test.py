"""Script to evalute performance on the val and test set."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transoar.utils.io import load_json, write_json
from transoar.data.dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator

class Tester:

    def __init__(self, args):
        path_to_run = Path(args.run)
        config = load_json(path_to_run / 'config.json')
        self._device = 'cuda:' + str(args.num_gpu)

        # Get path to checkpoint
        avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
        avail_checkpoints.sort(key=lambda x: len(str(x)))
        if args.last:
            path_to_ckpt = avail_checkpoints[0]
        else:
            path_to_ckpt = avail_checkpoints[-1]

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        try:
            self._path_to_results.mkdir(parents=True, exist_ok=False)
        except: 
            pass

        # Build necessary components
        self._set_to_eval = 'val' if args.val else 'test'
        self._test_loader = get_loader(config['data'], self._set_to_eval)

        self._evaluator = DetectionEvaluator(classes=list(config['data']['labels'].values()))
        self._model = TransoarNet(config['model'], config['data']['num_classes']).to(device=self._device)

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()

    def run(self):
        with torch.no_grad():
            for data, mask, bboxes, _ in tqdm(self._test_loader):
                # Put data to gpu
                data, mask = data.to(device=self._device), mask.to(device=self._device)
            
                targets = []
                for item in bboxes:
                    target = {
                        'boxes': item[0].to(dtype=torch.float, device=self._device),
                        'labels': item[1].to(device=self._device)
                    }
                    targets.append(target)

                # Make prediction
                out = self._model(data, mask)

                # Add pred to evaluator
                pred_probs = F.softmax(out['pred_logits'], dim=-1)
                self._evaluator.add(
                    pred_boxes=[boxes.detach().cpu().numpy() for boxes in out['pred_boxes']],
                    pred_classes=[torch.max(probs, dim=-1)[1].detach().cpu().numpy() for probs in pred_probs],
                    pred_scores=[torch.max(probs, dim=-1)[0].detach().cpu().numpy() for probs in pred_probs],
                    gt_boxes=[target['boxes'].detach().cpu().numpy() for target in targets],
                    gt_classes=[target['labels'].detach().cpu().numpy() for target in targets]
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
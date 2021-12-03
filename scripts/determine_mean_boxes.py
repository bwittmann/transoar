"""Script to determine the mean gt boxes for sanity check."""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

from transoar.utils.io import load_json
from transoar.data.dataloader import get_loader


def run(args):
    path_to_run = Path(args.run)
    config = load_json(path_to_run / 'config.json')

    set_to_eval = 'val'
    test_loader = get_loader(config, set_to_eval, batch_size=1)

    class_boxes = defaultdict(list)

    with torch.no_grad():
        for _, _, bboxes, _ in tqdm(test_loader):

            # if bboxes[0][1].shape[0] < 20:
            #     continue
            
            for box, classes in zip(bboxes[0][0], bboxes[0][1]):
                class_boxes[classes.item()].append(box[None])

    mean_boxes = []
    median_boxes = []
    for idx in range(1, 21):
       all_boxes = torch.cat(class_boxes[idx]) 
       mean_box = torch.mean(all_boxes, axis=0)
       median_box = torch.median(all_boxes, axis=0)[0]

       mean_boxes.append(mean_box[None])
       median_boxes.append(median_box[None])

    mean_boxes_tensor = torch.cat(mean_boxes, axis=0)
    median_boxes_tensor = torch.cat(median_boxes, axis=0)

    print(mean_boxes_tensor)
    print(median_boxes_tensor)

       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Path to experiment in transoar/runs.')
    args = parser.parse_args()
    run(args)

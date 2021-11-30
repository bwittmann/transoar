"""Inference script to infer final output from model prediction."""

import numpy as np
import torch
import torch.nn.functional as F

def inference(out):
    # Get probabilities from output logits
    pred_probs = F.softmax(out['pred_logits'], dim=-1)

    # Transform into np arrays and store as a list of arrays, as required in evaluator
    pred_boxes = [boxes.detach().cpu().numpy() for boxes in out['pred_boxes']]
    pred_classes = [torch.max(probs, dim=-1)[1].detach().cpu().numpy() for probs in pred_probs]
    pred_scores = [torch.max(probs, dim=-1)[0].detach().cpu().numpy() for probs in pred_probs]

    # Get rid of empty detections
    valid_ids = [np.nonzero(batch_elem_classes) for batch_elem_classes in pred_classes]
    pred_classes = [pred[ids] for pred, ids in zip(pred_classes, valid_ids)]
    pred_boxes = [pred[ids] for pred, ids in zip(pred_boxes, valid_ids)]
    pred_scores = [pred[ids] for pred, ids in zip(pred_scores, valid_ids)]

    # Get detection with highest score for each class as final prediction
    for idx, (batch_classes, batch_scores) in enumerate(zip(pred_classes, pred_scores)):
        max_ids = []
        unique_classes = np.unique(batch_classes)

        for class_ in unique_classes:
            class_idx = (batch_classes == class_).nonzero()[0]

            if class_idx.size > 1:
                class_scores = batch_scores[class_idx]
                max_ids.append(class_idx[class_scores.argmax()])
            else:
                max_ids.append(class_idx.item())

        pred_classes[idx] = pred_classes[idx][max_ids]
        pred_scores[idx] = pred_scores[idx][max_ids]
        pred_boxes[idx] = pred_boxes[idx][max_ids]

        assert pred_classes[idx].size <= 20

    return pred_boxes, pred_classes, pred_scores

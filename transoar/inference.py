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

    return pred_boxes, pred_classes, pred_scores

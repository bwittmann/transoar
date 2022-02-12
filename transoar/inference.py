"""Inference script to infer final output from model prediction."""

import numpy as np
import torch
import torch.nn.functional as F

def inference(out, query_info=None):
    # Get probabilities from output logits
    pred_probs = F.softmax(out['pred_logits'], dim=-1)
    classes_pred_probs = [torch.split(batch_probs, 27, dim=0) for batch_probs in pred_probs]
    classes_pred_boxes = [torch.split(batch_boxes, 27, dim=0) for batch_boxes in out['pred_boxes']]

    boxes = []
    classes = []
    scores = []
    for batch_classes_boxes, batch_classes_probs in zip(classes_pred_boxes, classes_pred_probs):
        batch_boxes = []
        batch_classes = []
        batch_scores = []
        for idx, (class_boxes, class_probs) in enumerate(zip(batch_classes_boxes, batch_classes_probs)):
            if class_probs[:, -1].max() > 0.5:
                batch_classes.append(idx + 1)
                batch_scores.append(class_probs[:, -1].max().detach().cpu().numpy())
                batch_boxes.append(class_boxes[class_probs[:, -1].argmax()][None].detach().cpu().numpy())

                if query_info is not None:
                    query_info[idx+1].append([class_probs[:, -1].max().item(), class_probs[:, -1].argmax().item()])

        boxes.append(np.concatenate(batch_boxes))
        classes.append(np.array(batch_classes))
        scores.append(np.array(batch_scores))

    if query_info is not None:
        return boxes, classes, scores, query_info
    else:
        return boxes, classes, scores

"""Inference script to infer final output from model prediction."""

import numpy as np

def inference(out, query_info=None):
    bs, *_ = out['pred_logits'].shape

    # Get probabilities from output logits and select query with highest prob
    pred_probs = out['pred_logits'].sigmoid().squeeze().reshape(bs, 20, 27)
    pred_boxes = out['pred_boxes'].reshape(bs, 20, 27, -1)
    pred_query_ids = pred_probs.argmax(dim=-1)

    # Adjust format to fit metric
    boxes = []
    classes = []
    scores = []
    for batch in range(bs):
        batch_boxes = []
        batch_classes = []
        batch_scores = []
        for class_ in range(20):
            valid_id = pred_query_ids[batch, class_]
            batch_boxes.append(pred_boxes[batch, class_, valid_id][None].detach().cpu().numpy())
            batch_scores.append(pred_probs[batch, class_, valid_id].detach().cpu().numpy())
            batch_classes.append(class_ + 1)

            if query_info is not None:
                query_info[class_ + 1].append(
                    [
                        pred_probs[batch, class_, valid_id].item(),
                        pred_query_ids[batch, class_].item()
                    ]
                )

        boxes.append(np.concatenate(batch_boxes))
        classes.append(np.array(batch_classes))
        scores.append(np.array(batch_scores))

    if query_info is not None:
        return boxes, classes, scores, query_info
    else:
        return boxes, classes, scores

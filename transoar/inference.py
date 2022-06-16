"""Inference script to infer final output from model prediction."""

import numpy as np

def inference(out, num_organs):
    bs, num_queries, _ = out['pred_logits'].shape
    num_queries_per_organ = int(num_queries / num_organs)

    # Get probabilities from output logits and select query with highest prob
    pred_probs = out['pred_logits'].sigmoid().squeeze().reshape(bs, num_organs, num_queries_per_organ).cpu()
    pred_boxes = out['pred_boxes'].reshape(bs, num_organs, num_queries_per_organ, -1).cpu()
    pred_query_ids = pred_probs.argmax(dim=-1)

    # Adjust format to fit metric
    boxes = []
    classes = []
    scores = []
    for batch in range(bs):
        batch_boxes = []
        batch_classes = []
        batch_scores = []
        for class_ in range(num_organs):
            valid_id = pred_query_ids[batch, class_]
            batch_boxes.append(pred_boxes[batch, class_, valid_id][None].detach().cpu().numpy())
            batch_scores.append(pred_probs[batch, class_, valid_id].detach().cpu().numpy())
            batch_classes.append(class_ + 1)


        boxes.append(np.concatenate(batch_boxes))
        classes.append(np.array(batch_classes))
        scores.append(np.array(batch_scores))

        return boxes, classes, scores

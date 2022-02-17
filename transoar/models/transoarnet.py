"""Main model of the transoar project."""

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._aux_loss = config['neck']['aux_loss'] # Use auxiliary decoding losses if required
        self._num_fmap = config['backbone']['num_fmap']

        # Get anchors        
        self.anchors = self._generate_anchors(config['neck'], config['bbox_properties']).cuda()

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get neck
        self._neck = build_neck(config['neck'], config['bbox_properties'])

        # Get heads
        hidden_dim = config['neck']['hidden_dim']
        self._cls_head = nn.Linear(hidden_dim, 1)
        self._bbox_reg_head = MLP(hidden_dim, hidden_dim, 6, 3)

        num_queries = config['neck']['num_queries']
        num_channels = config['backbone']['fpn_channels']
        self._query_embed = nn.Embedding(num_queries, hidden_dim * 2)   # 2 -> tgt + query_pos
        self._input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

        self._reset_parameter()

    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)

        nn.init.constant_(self._cls_head.weight.data, 0)
        nn.init.constant_(self._cls_head.bias.data, 0)


    def forward(self, x):
        out_backbone = self._backbone(x)

        src = self._input_proj(out_backbone[self._num_fmap])
        pos = self._pos_enc(src)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            src,
            self._query_embed.weight,
            pos
        )

        pred_logits = self._cls_head(out_neck)
        pred_boxes = self._bbox_reg_head(out_neck).tanh() * 0.2

        out = {
            'pred_logits': pred_logits[-1], # Take output of last layer
            'pred_boxes': pred_boxes[-1] + self.anchors
        }

        if self._aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_logits, pred_boxes)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_boxes):
        # Hack to support dictionary with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b + self.anchors}
                for a, b in zip(pred_logits[:-1], pred_boxes[:-1])]

    def _generate_anchors(self, model_config, bbox_props):
        median_bboxes = defaultdict(list)
        # Get median bbox for each class 
        for class_, class_bbox_props in bbox_props.items():
            median_bboxes[int(class_)] = class_bbox_props['median'] #cxcyczwhd

        # Init anchors and corresponding classes
        anchors = torch.zeros((model_config['num_queries'], 6))
        query_classes = torch.repeat_interleave(
            torch.arange(1, model_config['num_organs'] + 1), model_config['queries_per_organ'] * model_config['num_feature_levels']
        )

        # Generate offsets for each anchor
        anchor_offset = model_config['anchor_offsets']
        possible_offsets = torch.tensor([0, anchor_offset, -anchor_offset])
        offsets = torch.cartesian_prod(possible_offsets, possible_offsets, possible_offsets)
        offsets = offsets.repeat(model_config['num_organs'] * model_config['num_feature_levels'], 1)

        # Generate anchors by applying offsets to median box
        for idx, (query_class, offset) in enumerate(zip(query_classes, offsets)):
            query_median_box = torch.tensor(median_bboxes[query_class.item()])
            query_median_box[:3] += offset 
            anchors[idx] = query_median_box

        return anchors


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

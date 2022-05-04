"""Main model of the transoar project."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        self._input_level = config['neck']['input_level']
        self._anchor_offset = config['neck']['anchor_offset_pred']
        self._max_offset = config['neck']['max_anchor_pred_offset']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get anchors
        self._anchors = self._generate_anchors(config['neck'], config['bbox_properties']).cuda()

        # Get neck
        self._neck = build_neck(config['neck'], config['bbox_properties'])

        # Get heads
        self._cls_head = nn.Linear(hidden_dim, 1)
        #self._cls_head = MLP(hidden_dim, hidden_dim, 1, 3)
        self._bbox_reg_head = MLP(hidden_dim, hidden_dim, 6, 3)

        self._seg_proxy = config['backbone']['use_seg_proxy_loss']
        if self._seg_proxy:
            in_channels = config['backbone']['start_channels']
            out_channels = 2 if config['backbone']['fg_bg'] else config['neck']['num_organs'] + 1 # inc bg
            self._seg_head = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

        # Get projections and embeddings
        self._query_embed = nn.Embedding(num_queries, hidden_dim * 2)  # tgt + query_pos 

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

        if self._anchor_offset:
            self._reset_parameter()

    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)

        nn.init.constant_(self._cls_head.weight.data, 0)
        nn.init.constant_(self._cls_head.bias.data, 0)
        #nn.init.constant_(self._cls_head.layers[-1].weight.data, 0)
        #nn.init.constant_(self._cls_head.layers[-1].bias.data, 0)

    def _generate_anchors(self, model_config, bbox_props):
        # Get median bbox for each class 
        median_bboxes = defaultdict(list)
        for class_, class_bbox_props in bbox_props.items():
            median_bboxes[int(class_)] = class_bbox_props['median'] #cxcyczwhd

        # Init anchors and corresponding classes
        anchors = torch.zeros((model_config['num_queries'], 6))
        query_classes = torch.repeat_interleave(
            torch.arange(1, model_config['num_organs'] + 1), int(model_config['num_queries'] / model_config['num_organs'])
        )

        # Generate offsets for each anchor, possibly dynamic
        if not model_config['anchor_gen_dynamic_offset']:
            anchor_offset = model_config['anchor_gen_offset']
            possible_offsets = torch.tensor([0, anchor_offset, -anchor_offset])

            if anchors.shape[0] == 20:  # no offset
                offsets = torch.zeros_like(anchors[0][0:3][None])
            elif anchors.shape[0] == 140:   # 6 offsets
                all_offsets = torch.cartesian_prod(possible_offsets, possible_offsets, possible_offsets)
                offsets = all_offsets[torch.count_nonzero(all_offsets, dim=-1) <= 1]
            else:   # 26 offsets
                offsets = torch.cartesian_prod(possible_offsets, possible_offsets, possible_offsets)

            offsets = offsets.repeat(model_config['num_organs'], 1)
        else:
            offsets_combined = []
            for class_, class_bbox_props in bbox_props.items():
                attn_area = torch.tensor(class_bbox_props['attn_area']) # x1y1z1x2y2z2
                median_box = torch.tensor(class_bbox_props['median']) # cxcyczwhd
                offset_scale = (((attn_area[3:] - attn_area[0:3]) - median_box[3:]) / 3)[None]
                possible_offsets = torch.cat((offset_scale, -offset_scale, torch.zeros_like(offset_scale)), dim=0)

                if anchors.shape[0] == 20:  # no offset
                    offsets = torch.zeros_like(anchors[0][0:3][None])
                elif anchors.shape[0] == 140:   # 6 offsets
                    all_offsets = torch.cartesian_prod(*possible_offsets.unbind(dim=-1))
                    offsets = all_offsets[torch.count_nonzero(all_offsets, dim=-1) <= 1]
                else:   # 26 offsets
                    offsets = torch.cartesian_prod(*possible_offsets.unbind(dim=-1))
                offsets_combined.append(offsets)
            
            offsets = torch.cat(offsets_combined)

        # Generate anchors by applying offsets to median box
        for idx, (query_class, offset) in enumerate(zip(query_classes, offsets)):
            query_median_box = torch.tensor(median_bboxes[query_class.item()])
            query_median_box[:3] += offset 
            anchors[idx] = query_median_box

        return torch.clamp(anchors, min=0, max=1)

    def forward(self, x):
        out_backbone = self._backbone(x)
        seg_src = out_backbone['P0'] if self._seg_proxy else 0
        det_src = out_backbone[self._input_level]

        pos = self._pos_enc(det_src)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            det_src,
            self._query_embed.weight,
            pos
        )

        pred_logits = self._cls_head(out_neck)
        pred_boxes = self._bbox_reg_head(out_neck)
        if self._anchor_offset:
            pred_boxes = torch.clamp((pred_boxes.tanh() * self._max_offset) + self._anchors, min=0, max=1)
            # pred_boxes = (pred_boxes.tanh() * self._max_offset) + self._anchors
        else:
            pred_boxes =  pred_boxes.sigmoid()

        pred_seg = self._seg_head(seg_src) if self._seg_proxy else 0

        out = {
            'pred_logits': pred_logits[-1], # Take output of last layer
            'pred_boxes':pred_boxes[-1],
            'pred_seg': pred_seg
        }

        if self._aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_logits, pred_boxes)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_boxes):
        # Hack to support dictionary with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(pred_logits[:-1], pred_boxes[:-1])]

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

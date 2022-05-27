"""Main model of the transoar project."""

from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        self._input_levels = config['neck']['input_levels']
        self._anchor_offset = config['neck']['anchor_offset_pred']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get anchors
        anchors, restrictions = self._generate_anchors(config['neck'], config['bbox_properties'])
        self._anchors = anchors.cuda()
        self._restrictions = restrictions.cuda() if config['neck']['anchor_gen_dynamic_offset'] else config['neck']['max_anchor_pred_offset']

        # Get neck
        self._neck = build_neck(config['neck'], config['bbox_properties'], self._anchors, self._restrictions)

        # Get heads
        self._cls_head = nn.Linear(hidden_dim, 1)

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
        nn.init.constant_(self._cls_head.weight.data, 0)
        nn.init.constant_(self._cls_head.bias.data, 0)

        nn.init.constant_(self._neck.decoder.reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._neck.decoder.reg_head.layers[-1].bias.data, 0)

    def _generate_anchors(self, model_config, bbox_props):
        num_queries = model_config['num_queries']
        num_queries_per_organ = int(num_queries / model_config['num_organs'])

        def gen_offsets(pos_offsets, dyn=True):
            if dyn:
                all_offsets = torch.cartesian_prod(*pos_offsets.unbind(dim=-1))
            else:
                all_offsets = torch.cartesian_prod(pos_offsets, pos_offsets, pos_offsets)
            return all_offsets

        anchor_dict = defaultdict(dict)
        for class_, class_bbox_props in bbox_props.items():
            # Extract relevant info regarding size and pos
            median_size = torch.tensor(class_bbox_props['median'])[3:]  # whd

            attn_vol = torch.tensor(class_bbox_props['attn_area']) # x1y1z1x2y2z2
            attn_vol_center = (attn_vol[:3] + attn_vol[3:]) / 2 #cxcycz
            attn_vol_whd = attn_vol[3:] - attn_vol[:3]  # whd
            
            if model_config['anchor_gen_dynamic_offset']:
                pos_offsets = ((attn_vol_whd - median_size) / 2)[None]
                pos_offsets = torch.cat((pos_offsets, -pos_offsets, torch.zeros_like(pos_offsets)), dim=0)
            else:
                anchor_offset = model_config['anchor_gen_offset']
                pos_offsets = torch.tensor([0, anchor_offset, -anchor_offset]) 

            if num_queries == 20:  # no offset
                pos_offsets = torch.zeros(3)[None]
            elif num_queries == 140:   # 6 offsets
                all_offsets = gen_offsets(pos_offsets, model_config['anchor_gen_dynamic_offset'])
                pos_offsets = all_offsets[torch.count_nonzero(all_offsets, dim=-1) <= 1]
            else:   # 26 offsets
                pos_offsets = gen_offsets(pos_offsets, model_config['anchor_gen_dynamic_offset'])

            anchor_dict[class_]['pos_offsets'] = pos_offsets
            anchor_dict[class_]['base_center'] = attn_vol_center
            anchor_dict[class_]['base_size'] = median_size
            
        # Generate anchors by applying offsets to median box
        anchors = []
        restriction_pos_offsets = []
        for class_dict in anchor_dict.values():
            class_anchors = torch.cat((class_dict['pos_offsets'], class_dict['base_size'][None].repeat(class_dict['pos_offsets'].shape[0], 1)), dim=-1)
            class_anchors[:, :3] += class_dict['base_center']
            anchors.append(class_anchors)
            restriction_pos_offsets.append(class_dict['pos_offsets'].max(dim=0)[0][None])

        # Determine offset restrictions
        min_size_offsets = torch.tensor([v['median'] for v in bbox_props.values()])[:, 3:] - torch.tensor([v['min'] for v in bbox_props.values()])[:, 3:]
        max_size_offsets = torch.tensor([v['max'] for v in bbox_props.values()])[:, 3:] - torch.tensor([v['median'] for v in bbox_props.values()])[:, 3:]
        restriction_size_offsets = torch.max(min_size_offsets, max_size_offsets)   # whd
        restriction_pos_offsets = torch.cat(restriction_pos_offsets)
        restriction = torch.repeat_interleave(torch.cat((restriction_pos_offsets, restriction_size_offsets), dim=-1), num_queries_per_organ, dim=0)

        return torch.cat(anchors).clamp(min=0, max=1), restriction

    def forward(self, x):
        out_backbone = self._backbone(x)
        seg_src = out_backbone['P0'] if self._seg_proxy else 0
        det_src = out_backbone[self._input_levels]
        pos = self._pos_enc(det_src)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            det_src,
            self._query_embed.weight,
            pos
        )

        pred_logits = self._cls_head(out_neck)
        pred_boxes = self._neck.decoder.reg_head(out_neck)
        if self._anchor_offset:
            pred_boxes = torch.clamp((pred_boxes.tanh() * self._restrictions) + self._anchors, min=0, max=1)
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

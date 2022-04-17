"""Main model of the transoar project."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        num_classes = config['num_classes']
        self.input_level = config['neck']['input_level']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get neck
        self._neck = build_neck(config['neck'])

        # Get heads
        self._cls_head = nn.Linear(hidden_dim, num_classes + 1)
        self._bbox_reg_head = MLP(hidden_dim, hidden_dim, 6, 3)

        in_channels = config['backbone']['start_channels']
        out_channels = 2 if config['backbone']['fg_bg'] else config['neck']['num_organs'] + 1 # inc bg
        self._seg_head = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

        # Get projections and embeddings
        self._query_embed = nn.Embedding(num_queries, hidden_dim * 2)   # 2 -> tgt + query_pos

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

        self._reset_parameter()


    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data[2:], -2.0)


    def forward(self, x):
        out_backbone = self._backbone(x)
        seg_src = out_backbone['P0']

        # Retrieve fmaps
        det_srcs = []
        for key, value in out_backbone.items():
            if int(key[-1]) < int(self.input_level[-1]):
                continue
            else:
                det_srcs.append(value)

        det_masks = []
        det_pos = []
        for idx, fmap in enumerate(det_srcs):
            det_srcs[idx] = fmap
            mask = torch.zeros_like(fmap[:, 0], dtype=torch.bool)    # No mask needed
            det_masks.append(mask)
            det_pos.append(self._pos_enc(fmap))

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            det_srcs,
            det_masks,
            self._query_embed.weight,
            det_pos
        )

        # Relative offset box and logit prediction
        hs, init_reference_out, inter_references_out = out_neck

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference_out
            else:
                reference = inter_references_out[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self._cls_head(hs[lvl])
            tmp = self._bbox_reg_head(hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., :3] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        pred_logits = torch.stack(outputs_classes)
        pred_boxes = torch.stack(outputs_coords)

        # Segmentation prediction
        pred_seg = self._seg_head(seg_src)

        out = {
            'pred_logits': pred_logits[-1], # Take output of last layer
            'pred_boxes': pred_boxes[-1],
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

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

"""GNN for our knn graph."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, num_gnn_layers, p_drop):
        super().__init__()
        self.p_drop = p_drop

        # output proj
        self.readout = nn.Conv1d(h_dim, out_dim, kernel_size=1, stride=1)

        # gcn layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, h_dim))
        for _ in range(num_gnn_layers - 2):
            self.convs.append(SAGEConv(h_dim, h_dim))
        self.convs.append(SAGEConv(h_dim, h_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # for conv in self.convs[:-1]:
        #     x = conv(x, edge_index)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.p_drop, training=self.training)
        # x = self.convs[-1](x, edge_index)

        x = self.readout(x.T[None]).squeeze().T
        return x #F.log_softmax(x, dim=1)
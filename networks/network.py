#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResNet(nn.Module):
    def __init__(self,
                 opt,
                 input_dim=2,
                 index_dim=1,
                 hidden_dim=128,
                 n_hidden_layers=20):

        super().__init__()

        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers

        self.x_input = True # input is concat [x,v] or just x
        if self.x_input:
            in_dim = input_dim * 2 + index_dim 
        else:
            in_dim = input_dim + index_dim 
        out_dim = input_dim

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim + index_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim + index_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.layers[-1] = zero_module(self.layers[-1])
    def _append_time(self, h, t):
        time_embedding = torch.log(t)
        return torch.cat([h, time_embedding.reshape(-1, 1)], dim=1)

    def forward(self, u, t,class_labels=None):
        h0 = self.layers[0](self._append_time(u, t))
        h = self.act(h0)

        for i in range(self.n_hidden_layers):
            h_new = self.layers[i + 1](self._append_time(h, t))
            h = self.act(h + h_new)

        return self.layers[-1](self._append_time(h, t))

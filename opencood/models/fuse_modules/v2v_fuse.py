# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn

from opencood.models.sub_modules.CoE2E_utils import LayerNorm
from opencood.models.sub_modules.convgru import ConvGRU


class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']

        self.pre_norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")

        self.msg_cnn = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
                                     LayerNorm(in_channels, eps=1e-6, data_format="channels_first"))
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Sequential(nn.Linear(in_channels, in_channels),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Dropout(0.05))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, sttf_mask):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (B, C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        L, C, H, W = x.shape
        x = self.pre_norm(x)

        for l in range(self.num_iteration):
            updated_node_features = []
            for agent in range(L):
                ego_agent_feature = x[agent].unsqueeze(0).repeat(L, 1, 1, 1)
                neighbor_feature = torch.cat([x, ego_agent_feature], dim=1)

                message = self.msg_cnn(neighbor_feature) * sttf_mask
                agg_feature, _ = torch.max(message, dim=0)

                cat_feature = torch.cat([x[agent], agg_feature], dim=0)
                if self.gru_flag:
                    gru_out = self.conv_gru(cat_feature.unsqueeze(0).unsqueeze(0))[0][0].squeeze(0).squeeze(0)
                else:
                    gru_out = x[agent] + agg_feature

                updated_node_features.append(gru_out.unsqueeze(0))

            x = torch.cat(updated_node_features, dim=0)

        out = self.mlp(x[0].permute(1, 2, 0)).permute(2, 0, 1)

        return out

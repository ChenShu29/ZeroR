import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.CoE2E_utils import LayerNorm, SELayer


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == len(self.model_cfg['layer_strides']) == len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x


class ASPPModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rates=(6, 12)):
        super().__init__()
        self.convs = nn.ModuleList()
        for r in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 3, padding=r, dilation=r, bias=False),
                LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
                nn.ReLU()
            ))
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*hidden_dim + in_dim, out_dim, 1, bias=False),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
            nn.ReLU()
        )

    def forward(self, x):
        res = [x]
        for conv in self.convs:
            res.append(conv(x))
        out = torch.cat(res, dim=1)
        return self.project(out)


class BevSegHead(nn.Module):
    # TODO: 原工作没有同时做动态和静态分割，两者的整个模型参数都不同
    def __init__(self, target, input_dim, output_class_dynamic, output_class_static):
        super(BevSegHead, self).__init__()
        self.target = target
        if self.target == 'dynamic':
            self.dynamic_head = nn.Conv2d(input_dim, output_class_dynamic, kernel_size=3, padding=1)
        if self.target == 'static':
            # segmentation head
            self.static_head = nn.Conv2d(input_dim, output_class_static, kernel_size=3, padding=1)
        else:
            self.dynamic_head = nn.Conv2d(input_dim, output_class_dynamic, kernel_size=3, padding=1)
            self.static_head = nn.Conv2d(input_dim, output_class_static, kernel_size=3, padding=1)

    def forward(self, x):
        if self.target == 'dynamic':
            dynamic_map = self.dynamic_head(x)
            static_map = torch.zeros_like(dynamic_map, device=dynamic_map.device)

        elif self.target == 'static':
            static_map = self.static_head(x)
            dynamic_map = torch.zeros_like(static_map, device=static_map.device)

        else:
            dynamic_map = self.dynamic_head(x)
            static_map = self.static_head(x)

        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        return output_dict


class BevSegHeadEnhance(nn.Module):
    # TODO: 原工作没有同时做动态和静态分割，两者的整个模型参数都不同
    def __init__(self, input_dim, output_class):
        super(BevSegHeadEnhance, self).__init__()

        self.se_layer = SELayer(channel=input_dim, reduction=4)

        self.ASPP = ASPPModule(input_dim, input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(input_dim, output_class, 3, padding=1)
        )

    def forward(self, x):
        x = self.se_layer(x)
        x = self.ASPP(x)
        x = self.classifier(x)

        return x
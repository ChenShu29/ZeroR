import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.CoE2E_utils import LayerNorm


# For Spatial Refinement
class PyramidPool(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPool, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self, x, spatial_channel_mask):
        """
        :param x: Multi-agent features each batch [L, C, H, W]
        :param spatial_channel_mask: [L, ]
        :return:
        """
        #   ->  [1, C, H, W]

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)

        conv1 = F.interpolate(conv1, size=x.size()[2:], mode='bilinear', align_corners=False)
        conv2 = F.interpolate(conv2, size=x.size()[2:], mode='bilinear', align_corners=False)
        conv3 = F.interpolate(conv3, size=x.size()[2:], mode='bilinear', align_corners=False)

        fused = torch.cat([conv1, conv2, conv3], dim=1)
        fused = self.fuse(fused)
        return fused


# For Spatial Refinement
class MultiScaleDilaConv(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleDilaConv, self).__init__()
        down_dim_half = in_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, down_dim_half, kernel_size=1),
            LayerNorm(down_dim_half, eps=1e-6, data_format="channels_first"),
            nn.PReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, down_dim_half, kernel_size=3, dilation=2, padding=2),
            LayerNorm(down_dim_half, eps=1e-6, data_format="channels_first"),
            nn.PReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, down_dim_half, kernel_size=3, dilation=4, padding=4),
            LayerNorm(down_dim_half, eps=1e-6, data_format="channels_first"),
            nn.PReLU())

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * down_dim_half, in_channels, kernel_size=1),
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.PReLU())

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)

        fused = torch.cat([conv1, conv2, conv3], dim=1)
        fused, _ = (self.fuse(fused) + x).max(dim=0)
        return fused


# For Spatial Refinement
class DynamicConvFusion(nn.Module):
    """Dynamically generate convolutional kernel based on features"""
    def __init__(self, n_agents, channels, kernel_size=3):
        super().__init__()
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(n_agents * channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, n_agents * kernel_size ** 2, 1)
        )
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):  # x: [B, n, C, H, W]
        B, n, C, H, W = x.shape
        x_stack = x.view(B, n * C, H, W)

        kernel_weights = self.kernel_gen(x_stack)  # [B, n*k^2, H, W]
        kernel_weights = kernel_weights.view(B, n, self.kernel_size ** 2, H, W)

        output = 0
        for i in range(n):
            agent_feat = x[:, i]  # [B, C, H, W]
            kernels = kernel_weights[:, i]  # [B, k^2, H, W]

            unfolded = nn.functional.unfold(agent_feat, self.kernel_size, padding=self.padding)  # [B, C*k^2, H*W]
            unfolded = unfolded.view(B, C, self.kernel_size ** 2, H, W)

            conv_result = (unfolded * kernels.unsqueeze(1)).sum(dim=2)  # [B, C, H, W]
            output += conv_result

        return output / n

# For multi-Agent Feature Fusion
class SENet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_weights = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2),
            nn.ReLU(),
            nn.Softmax(dim=0))

    def forward(self, x):  # x: [L, C, H, W]
        channel_weights = self.channel_weights(x)

        weighted = (x * channel_weights).sum(dim=0)  # [B, C, H, W]

        return weighted


class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y


class DCAB(nn.Module):
    """Modified from https://github.com/ICSResearch/CPP-Net"""
    def __init__(self, dim, reduction):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
            CA(dim, reduction),
        )

    def forward(self, x):
        x = self.block(x) + x
        return x
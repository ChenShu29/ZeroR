import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from einops import rearrange
from opencood.models.sub_modules.CoE2E_utils import LayerNorm, SELayer


class ResnetEncoder(nn.Module):
    """
    Resnet family to encode image.

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """

    def __init__(self, params):
        super(ResnetEncoder, self).__init__()

        self.num_layers = params['num_layers']
        self.pretrained = params['pretrained']
        image_height = params['image_height']
        image_width = params['image_width']
        self.idx_pick = params['id_pick']

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet "
                "layers".format(self.num_layers))

        self.encoder = resnets[self.num_layers](weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 1, image_height, image_width, 3)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, input_images):
        """
        Compute deep features from input images.

        Parameters
        ----------
        input_images : torch.Tensor
            The input images have shape of (B,L,M,H,W,3), where L, M are
            the num of agents and num of cameras per agents.

        Returns
        -------
        features: torch.Tensor
            The deep features for each image with a shape of (B,L,M,C,H,W)
        """
        b, v, h, w, c = input_images.shape
        input_images = input_images.view(b * v, h, w, c).contiguous()
        # b, h, w, c -> b, c, h, w
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(input_images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x0 = self.encoder.layer1(self.encoder.maxpool(x))
        x1 = self.encoder.layer2(x0)
        x2 = self.encoder.layer3(x1)
        x3 = self.encoder.layer4(x2)

        x0 = rearrange(x0, '(b v) c h w -> b v c h w', b=b, v=v)
        x1 = rearrange(x1, '(b v) c h w -> b v c h w', b=b, v=v)
        x2 = rearrange(x2, '(b v) c h w -> b v c h w', b=b, v=v)
        x3 = rearrange(x3, '(b v) c h w -> b v c h w', b=b, v=v)
        results = [x0, x1, x2, x3]

        if isinstance(self.idx_pick, list):
            return [results[i] for i in self.idx_pick]
        else:
            return results[self.idx_pick]



class NaiveDecoder(nn.Module):
    """
    A Naive decoder implementation

    Parameters
    ----------
    params: dict

    Attributes
    ----------
    num_ch_dec : list
        The decoder layer channel numbers.

    num_layer : int
        The number of decoder layers.

    input_dim : int
        The channel number of the input to
    """
    def __init__(self, params):
        super(NaiveDecoder, self).__init__()

        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']

        assert len(self.num_ch_dec) == self.num_layer

        self.se_layer = SELayer(channel=self.input_dim, reduction=4)

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layer-1, -1, -1):
            # upconv_0
            num_ch_in = self.input_dim if i == self.num_layer-1\
                else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = LayerNorm(normalized_shape=num_ch_out, eps=1e-5, data_format='channels_first')
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = LayerNorm(normalized_shape=num_ch_out, eps=1e-5, data_format='channels_first')
            self.convs[("relu", i, 1)] = nn.ReLU(True)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def upsample(x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x):
        """
        Upsample to

        Parameters
        ----------
        x : torch.tensor
            The bev bottleneck feature, shape: (B, L, C1, H, W)

        Returns
        -------
        Output features with (B, L, C2, H, W)
        """
        x = self.se_layer(x)

        for i in range(self.num_layer-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)

            x = self.upsample(x)

            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)
            x = self.convs[("relu", i, 1)](x)

        return x


class CVTDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
        super().__init__()

        dim = out_channels // factor

        if upsample:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                LayerNorm(normalized_shape=dim, eps=1e-5, data_format='channels_first'),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                LayerNorm(normalized_shape=out_channels, eps=1e-5, data_format='channels_first'))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
                LayerNorm(normalized_shape=dim, eps=1e-5, data_format='channels_first'),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
                LayerNorm(normalized_shape=out_channels, eps=1e-5, data_format='channels_first'))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up
        if self.with_relu:
            return self.relu(x)
        return x


class CVTDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args['input_dim']
        blocks = args['blocks']
        residual = args['residual']
        factor = args['factor']
        upsample = args['upsample']

        layers = []
        channels = input_dim


        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1  # if not last block, with relu
            layer = CVTDecoderBlock(channels, out_channels, input_dim, residual, factor, upsample, with_relu=with_relu)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.se_layer(x)
        y = x
        for layer in self.layers:
            y = layer(y, x)
        return y
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from einops import rearrange
from opencood.models.fuse_modules.fuse_utils import batch_merge
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_discretized_transformation_matrix

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-3, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix, record_len, previous_trans=None):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego
        record_len:
        previous_trans:

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        # [B, Max_cav, 4, 4] -> [L, 4, 4]
        spatial_correction_matrix = batch_merge(spatial_correction_matrix, record_len)
        if previous_trans is not None:
            spatial_correction_matrix = torch.cat([previous_trans, spatial_correction_matrix], dim=0)
        # [L, 2, 3]
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio, self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b c h w  -> b c w h')
        x = torch.flip(x, dims=(3,))
        # Only compensate non-ego vehicles
        B, C, H, W = x.shape


        T = get_transformation_matrix(dist_correction_matrix, (H, W))
        x = warp_affine(x, T,(H, W))

        # flip and transpose back
        x = torch.flip(x, dims=(3,))
        x = rearrange(x, 'b c w h -> b c h w')

        if previous_trans is not None:
            return x, spatial_correction_matrix
        else:
            return x

class OccReserve:

    @staticmethod
    def initialize(reserve_len, first_occ):

        return first_occ.unsqueeze(0).repeat(reserve_len, 1, 1, 1)

    @staticmethod
    def update(reserved_occ, current_occ, ego_motion_trans):
        # reserved length, channel (normally 1), Height, Weight
        L, C, H, W = reserved_occ.shape

        reserved_occ = rearrange(reserved_occ, 'l c h w  -> l c w h')
        reserved_occ = torch.flip(reserved_occ, dims=(3,))

        # Calculate transformation matrix to correct ego motion (2D)
        ego_motion_trans = ego_motion_trans.unsqueeze(0)
        dist_correction_matrix = get_discretized_transformation_matrix(
            matrix=ego_motion_trans, discrete_ratio=0.390625, downsample_rate=1)
        T = get_transformation_matrix(dist_correction_matrix, (H, W))
        T = T.repeat(L, 1, 1)

        # [L, C, W, H]
        reserved_occ = warp_affine(reserved_occ, T, (H, W))

        reserved_occ = torch.flip(reserved_occ, dims=(3,))

        reserved_occ = rearrange(reserved_occ, 'l c w h -> l c h w')


        return torch.cat((reserved_occ[1:], current_occ.unsqueeze(0)), dim=0)


class PositionalEncoding2D(nn.Module):
    "Modified from https://github.com/tatp22/multidim-positional-encoding"

    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 4) * 2)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2, device=tensor.device).float() / self.channels))
        # inv_freq = torch.tensor(inv_freq, device=tensor.device)
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        emb_x = torch.flatten(torch.stack((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1), -2, -1).unsqueeze(1)
        emb_y = torch.flatten(torch.stack((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1), -2, -1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device,
                          dtype=(self.dtype_override if self.dtype_override is not None else tensor.dtype), )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb.unsqueeze(0).repeat(tensor.shape[0], 1, 1, 1)

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.row_embed.weight)
        nn.init.xavier_uniform_(self.col_embed.weight)

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                        dim=-1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class LearnedPositionalEncoding3D(nn.Module):
    """Position embedding with learnable embedding weights."""
    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, temporal_embed=5):
        super(LearnedPositionalEncoding3D, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.tem_embed = nn.Embedding(temporal_embed, num_feats - 2)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.row_embed.weight)
        nn.init.xavier_uniform_(self.col_embed.weight)
        nn.init.xavier_uniform_(self.tem_embed.weight)

    def forward(self, mask_shape, device):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        t, h, w = mask_shape
        z = torch.arange(t, device=device)
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        z_embed = self.tem_embed(z).view(t, 1, 1, self.num_feats-2)
        x_embed = self.col_embed(x).view(1, 1, w, self.num_feats)
        y_embed = self.row_embed(y).view(1, h, 1, self.num_feats)

        pos = torch.cat([y_embed.repeat(t, 1, w, 1), x_embed.repeat(t, h, 1, 1), z_embed.repeat(1, h, w, 1)], dim=-1)

        return pos.unsqueeze(0)


def get_reference_points(H, W, bs, device='cuda', dtype=torch.float):
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                  torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device), indexing='ij')
    # Normalization for better training - It will be re-scale to feature size by grid_sample
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_2d = torch.stack((ref_x, ref_y), -1)
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    return ref_2d


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.gelu = F.gelu
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.gelu(self.norm(self.proj(x)))
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.01, projection_dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.25

        self.query = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.pre_norm_query = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.gelu = F.gelu
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, query, value):
        B, N, C = value.shape
        head_dim = C // self.num_heads
        query = self.query(self.pre_norm_query(query)).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        kv = self.kv(value).reshape(B, N, self.num_heads, head_dim * 2).permute(0, 2, 1, 3)
        k, v = kv[..., :head_dim], kv[..., head_dim:]

        attn_score = (query @ k.transpose(-2, -1)) * self.scale
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.attn_drop(attn_score)

        output = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        output = self.gelu(self.norm(self.proj(output)))
        output = self.proj_drop(output) + value

        return output


class LocalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, window_size):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.kH = window_size[0]
        self.kW = window_size[1]

    @staticmethod
    def f_similar(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = x_theta.permute(0, 2, 3, 1).contiguous()
        x_theta = x_theta.view(n * h * w, 1, c)

        x_phi = F.unfold(x_phi, kernel_size=(kh, kw), stride=1, padding=pad)
        x_phi = x_phi.contiguous().view(n, c, kh * kw, h * w)
        x_phi = x_phi.permute(0, 3, 1, 2).contiguous()
        x_phi = x_phi.view(n * h * w, c, kh * kw)

        out = torch.matmul(x_theta, x_phi)
        out = out.view(n, h, w, kh * kw)

        return out

    @staticmethod
    def f_weighting(x_theta, x_phi, kh, kw):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad)
        x_theta = x_theta.permute(0, 2, 1).contiguous()
        x_theta = x_theta.view(n * h * w, c, kh * kw)

        x_phi = x_phi.view(n * h * w, kh * kw, 1)

        out = torch.matmul(x_theta, x_phi)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        weight = self.f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(x3, weight, self.kH, self.kW)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x 


def dense_interpolation(feature, mask, scale_align="False"):
    l, c, h, w = feature.shape

    if scale_align == "True":
        feature_valid = feature.permute(0, 2, 3, 1).reshape(-1, c)
        mask_valid = mask.flatten()
        mask_valid = torch.nonzero(mask_valid, as_tuple=False).squeeze()
        feature_valid = feature_valid[mask_valid, :]

        global_mu = feature_valid.mean()
        global_std = feature_valid.std() + 1e-6
        noise = torch.randn_like(feature) * global_std + global_mu
        noise = torch.clamp(noise, feature_valid.min(), feature_valid.max()) * (1 - mask)

    else:
        noise = torch.randn_like(feature) * (1 - mask)

    init_guess = noise + feature
    init_guess = F.interpolate(init_guess, scale_factor=2, mode='bilinear', align_corners=False)
    init_guess = F.interpolate(init_guess, size=[h, w], mode='bilinear', align_corners=False)

    return feature + init_guess * (1 - mask)


def communication_cal(spatial_mask, sttf_mask, channels):
    """
    :return: the communication volume for each CAV
    """

    l, h, w, _ = sttf_mask.shape

    assert l > 1, "No collaborator, no communication happen"
    if spatial_mask is not None:
        assert l == spatial_mask.shape[0], "Collaborator number errors"

    if spatial_mask is not None:
        # selected_cell_num = torch.sum(spatial_mask[1:] * sttf_mask[1:])
        selected_cell_num = torch.sum(spatial_mask[1:])
    else:
        selected_cell_num = torch.sum(sttf_mask[1:])

    comm_rate = selected_cell_num / (h * w * (l-1))

    if comm_rate < 0:
        comm_rate = 0

    comm_volume = (selected_cell_num * channels * 32 / 8) / (l-1)

    if comm_volume < 0 or torch.isinf(comm_volume):
        comm_volume = 256

    return comm_rate, comm_volume


def spatial_mask_combine(sttf_mask, spatial_mask):

    combine_mask = sttf_mask[1:] * spatial_mask[1:]

    combine_mask_all = sttf_mask.clone()

    combine_mask_all[1:] = combine_mask

    return combine_mask_all

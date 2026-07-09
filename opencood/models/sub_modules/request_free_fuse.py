import torch
import torch.nn as nn
import random
import math

from opencood.models.sub_modules.CoE2E_utils import PositionalEncoding2D, Attention, CrossAttention, get_reference_points, LayerNorm
from opencood.MSDA_utils.modules.ms_deform_attn import MSDeformAttn


def pose_correlations(trans_matrix, h, w):
    with torch.no_grad():
        L = trans_matrix.shape[0]
        # generate initial grid with the same 2D size of BEV
        grid_y, grid_x = torch.meshgrid(torch.linspace(0.5, h - 0.5, h).cuda(), torch.linspace(0.5, w - 0.5, w).cuda(), indexing='ij')
        grid_init = torch.stack([grid_x / h, grid_y / w, torch.zeros_like(grid_x), torch.ones_like(grid_x)], dim=-1).repeat(L, 1, 1, 1)

        # Simulate the transformation correlations between collaborators and ego
        grid_t = torch.einsum('bij,bhwj->bhwi', trans_matrix, grid_init)[..., :3]
        dist = grid_t - grid_init[..., :3]
        return dist


def sampling_points_identify(sampling_offset, h, w, l):
    """
    Generate 2D reference points and conduct the offset, returning coordinates after offset
    :param sampling_offset: [L, H*W, n, 2]     ->  offset coordinates
    :return: -> [l, H*W, num_points, 2]      ->  sampled points' coordinate after offset
    """
    with torch.no_grad():
        # Get reference points
        reference_points = get_reference_points(H=h, W=w, bs=l)

        # Normalize offsets to a region [-3, 3]
        offset_min = sampling_offset.reshape(-1, 2).min(dim=0)[0]
        offset_max = sampling_offset.reshape(-1, 2).max(dim=0)[0]
        sampling_offset = (sampling_offset - offset_min) / (offset_max - offset_min + 1e-6)
        sampling_offset = sampling_offset * 4 - 2
        sampling_offset = torch.clamp(sampling_offset, min=-2, max=2)

        # Update reference points coordinate
        sampling_points = torch.floor(reference_points + sampling_offset)
        sampling_points[..., 0] = torch.clamp(sampling_points[..., 0], min=0, max=h-1)
        sampling_points[..., 1] = torch.clamp(sampling_points[..., 1], min=0, max=w-1)

        # -> [l, H*W, num_points, 2]
        return sampling_points


def spatial_masking_with_sampled(spatial_mask, sampling_points, mask_ratio):
    l, h, w, _ = spatial_mask.shape
    k = int(h * w * mask_ratio)

    # Get top k feature pixels -> [B, H*W] -> indices [B, k]
    spatial_mask = spatial_mask.flatten(1, 3)
    _, top_k_indices = torch.topk(spatial_mask, k=k, dim=1, largest=True, sorted=False)

    # Generate binary mask
    binary_mask = torch.zeros_like(spatial_mask)
    batch_indices = torch.arange(l, device=spatial_mask.device)[:, None].expand(-1, k)

    # Extract sampled points of these top k points    [L, H*W, num_points, 2] -> [L, k, num_points, 2] -> [L, k * num_points, 2]
    sampling_points = sampling_points[batch_indices, top_k_indices].flatten(1, 2)
    sampling_points = sampling_points[..., 0] * w + sampling_points[..., 1]

    # get all selected points indices [L, k + k*num_points]
    top_k_indices = torch.cat([top_k_indices, sampling_points], dim=-1).to(torch.int64)
    batch_indices = torch.arange(l, device=spatial_mask.device)[:, None].expand(-1, top_k_indices.shape[-1])

    # Masking operation
    binary_mask[batch_indices, top_k_indices] = 1.0
    binary_mask = binary_mask.view(l, h, w).unsqueeze(1)

    return binary_mask


def spatial_masking_normal(spatial_mask, mask_ratio):
    l, h, w, _ = spatial_mask.shape
    k = int(h * w * mask_ratio)

    # Get top k feature pixels -> [B, H*W] -> indices [B, k]
    spatial_mask = spatial_mask.flatten(1, 3)
    _, top_k_indices = torch.topk(spatial_mask, k=k, dim=1, largest=True, sorted=False)

    # Generate binary mask
    binary_mask = torch.zeros_like(spatial_mask)
    batch_indices = torch.arange(l, device=spatial_mask.device)[:, None].expand(-1, k)

    # Masking operation
    binary_mask[batch_indices, top_k_indices] = 1.0
    binary_mask = binary_mask.view(l, h, w).unsqueeze(-1)

    return binary_mask, top_k_indices


class RequestFreeSelection(nn.Module):
    def __init__(self, inter_dim, offset_points, num_heads=8):
        super(RequestFreeSelection, self).__init__()

        self.offset_points = offset_points
        self.num_heads = num_heads
        self.dim = inter_dim

        self.linear_spatial = nn.Linear(in_features=inter_dim // 2 + 4, out_features=inter_dim // 2)
        self.relu_spatial = nn.LeakyReLU(negative_slope=0.1)
        self.linear_feat = nn.Linear(in_features=inter_dim, out_features=inter_dim // 2)
        self.relu_feat = nn.LeakyReLU(negative_slope=0.1)

        self.pos_embedding = nn.Embedding(num_embeddings=1024, embedding_dim= inter_dim // 2)
        self.pos_encoding = PositionalEncoding2D(channels=inter_dim // 2)
        self.layer_norm = LayerNorm(3, eps=1e-6, data_format="channels_last")

        self.feat_norm = LayerNorm(inter_dim, eps=1e-6, data_format="channels_last")
        self.self_attn = Attention(dim=inter_dim, num_heads=num_heads, attention_dropout=0.05, projection_dropout=0.05)
        self.cross_attn = CrossAttention(dim=inter_dim, num_heads=num_heads, attention_dropout=0.05, projection_dropout=0.05)

        self.mask_norm = LayerNorm(inter_dim * 2, eps=1e-6, data_format="channels_last")
        self.projection_1 = nn.Sequential(
            nn.Linear(in_features=inter_dim * 2, out_features=inter_dim),
            nn.Dropout(0.05),
            nn.Linear(in_features=inter_dim, out_features=inter_dim // 2),
            nn.LeakyReLU(negative_slope=0.1))

        self.projection_2 = nn.Sequential(
            nn.Conv2d(inter_dim // 2, inter_dim // 4, kernel_size=3, padding=1),
            nn.Dropout(0.05),
            nn.Conv2d(inter_dim // 4, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1))

        self.sampling_offsets = nn.Sequential(
            nn.Linear(in_features=inter_dim * 2, out_features=inter_dim),
            nn.Dropout(0.05),
            nn.Linear(in_features=inter_dim, out_features=num_heads * offset_points * 2),
            nn.LeakyReLU(negative_slope=0.1))

        self.deform_query = nn.Sequential(
            nn.Linear(in_features=inter_dim * 2, out_features=inter_dim),
            nn.Dropout(0.05),
            nn.Linear(in_features=inter_dim, out_features=num_heads * offset_points),
            nn.LeakyReLU(negative_slope=0.1))

        self.ms_deform_attn = MSDeformAttn(d_model=inter_dim, n_heads=num_heads, offset_dim=2)

        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets[2].weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(
            1, 1, self.offset_points, 1)

        for i in range(self.offset_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets[2].bias.data = grid_init.view(-1)
        nn.init.constant_(self.deform_query[2].weight.data, 0.)
        nn.init.constant_(self.deform_query[2].bias.data, 0.)

        nn.init.kaiming_normal_(self.linear_spatial.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_feat.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.cross_attn.query.weight, gain=2.0)

    def spatial_embedding(self, transformation_matrix, h, w):
        l = transformation_matrix.shape[0]

        dist = pose_correlations(transformation_matrix, h, w)
        dist = self.layer_norm(dist)

        # Positional Encoding (positional-aware)
        pos_encoding = self.pos_embedding.weight.unsqueeze(0).repeat(l, 1, 1).reshape(l, h, w, -1)
        pos_encoding = self.pos_encoding(pos_encoding)

        # All spatial-wise embedding
        spatial_embed = torch.cat([pos_encoding, dist], dim=-1)

        return spatial_embed


    def forward(self, all_features_2d, spatial_embed, sttf_mask):
        l, h, w, _ = spatial_embed.shape
        region = torch.tensor([h, w], device=spatial_embed.device)

        # [L, C, H, W]  ->  [L, H, W, C]
        all_features_2d = self.feat_norm(all_features_2d.permute(0, 2, 3, 1))
        # features_non_ego = (all_features_2d.clone() * sttf_mask).flatten(-2, -1).permute(0, 2, 1)

        spatial_embed = self.relu_spatial(self.linear_spatial(torch.cat([spatial_embed, sttf_mask], dim=-1)))
        feat_embed = self.relu_feat(self.linear_feat(all_features_2d))
        spatial_mask = self.cross_attn(torch.cat([spatial_embed, feat_embed], dim=-1).flatten(-3, -2), all_features_2d.flatten(-3, -2)).reshape(l, h, w, -1)

        # Non-ego feature-based mask (Self-Attention)[
        # [L-1, H*W, dim] -> [L-1, H, W, dim]
        features_non_ego = self.self_attn(all_features_2d.flatten(-3, -2)).reshape(l, h, w, -1) + all_features_2d

        # Mask and sampling offset generation
        spatial_mask = self.mask_norm(torch.cat([spatial_mask, features_non_ego], dim=-1))
        attention_mask_map = self.projection_1(spatial_mask).permute(0, 3, 1, 2)
        attention_mask_map = self.projection_2(attention_mask_map).permute(0, 2, 3, 1)
        # attention_mask_map = torch.sigmoid(attention_mask_map)

        # Feature Selection
        mask_ratio = 0.32     # fix for inference
        spatial_mask_map, top_k_indices = spatial_masking_normal(attention_mask_map, mask_ratio)
        # Get attention weights - [L, H*W, n] -> [L, H*W, num_head, n]
        attention_weights = self.deform_query(spatial_mask.flatten(1, 2)).reshape(l, h*w, self.num_heads, self.offset_points)
        attention_weights = attention_weights.softmax(-1).unsqueeze(-2)

        # offset generation and get sampling locations    reference_points-[L, H*W, 1, 2], offsets-[L, H*W, n, 2]
        offsets = self.sampling_offsets(spatial_mask).flatten(1, 2).reshape(l, h*w, self.num_heads, self.offset_points, 2)
        reference_points = get_reference_points(H=h, W=w, bs=l)
        reference_points = (reference_points[:, :, None, :, :] + offsets / region).unsqueeze(-3)

        features_non_ego = features_non_ego.flatten(1, 2).reshape(l, h*w, self.num_heads, -1)

        features_non_ego = self.ms_deform_attn(features_non_ego, region, reference_points, attention_weights)

        features_non_ego = features_non_ego.reshape(l, h, w, -1)
        feature_ego = all_features_2d[0:1] + features_non_ego[0:1]
        features_non_ego = (features_non_ego[1:] + all_features_2d[1:]) * spatial_mask_map[1:]

        return torch.cat([feature_ego, features_non_ego], dim=0).permute(0, 3, 1, 2), spatial_mask_map
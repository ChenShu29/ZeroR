import math
import random
import torch
import torch.nn as nn


class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc-256'):
        super().__init__()
        self.topk_ratio = topk_ratio
        # print(self.topk_ratio)
        # self.topk_ratio = random.uniform(0.05,0.25)
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-32':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 32, 1),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.v_proj = nn.Linear(input_dim, input_dim)

    def forward(self, src, pos_embed, dis_priority):

        bs, c, h, w = src.shape
        # 各位置的分数
        src_dis = dis_priority * src.permute(1, 0, 2, 3)
        src_dis = src_dis.permute(1, 0, 2, 3).float()
        # print(src_dis.shape)
        sample_weight = self.score_pred_net(src_dis).sigmoid().view(bs, -1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()

        if self.training:
            sample_ratio = random.uniform(0.05, 0.55)
        else:
            sample_ratio = self.topk_ratio

        ##max sample number:
        sample_lens = torch.tensor(h * w * sample_ratio).repeat(bs, 1).int()
        max_sample_num = sample_lens.max()

        min_sample_num = sample_lens.min()
        sort_order = sample_weight_clone.sort(descending=True, dim=1)[1]
        sort_confidence_topk = sort_order[:, :max_sample_num]
        sort_confidence_topk_remaining = sort_order[:, min_sample_num:]
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        sample_reg_loss = sample_weight.gather(1, sort_confidence_topk).mean()
        src_sampled = src.gather(0, sort_confidence_topk.permute(1, 0)[..., None].expand(-1, -1,
                                                                                         c)) * sample_weight.gather(1,
                                                                                                                    sort_confidence_topk).permute(
            1, 0).unsqueeze(-1)
        # pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        pos_embed_sampled = pos_embed.gather(0, sort_confidence_topk.permute(1, 0)[..., None])

        return src_sampled, sample_reg_loss, sort_confidence_topk, pos_embed_sampled


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def merge_tokens(x, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """
    B, N, C = x.shape
    #B,N
    idx_token = torch.arange(N)[None, :].repeat(B, 1)
    # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    agg_weight = x.new_ones(B, N, 1)
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)
    #[[0]]
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    return x_merged,idx_token_new


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k=5):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        B, N, C = x.shape
        # print(x.shape)
        # exit()
        #批量计算两个向量集合的距离(默认为欧氏距离)
        #B,N,N
        dist_matrix = torch.cdist(x, x) / (C ** 0.5)
        # print(dist_matrix.shape)
        # get local density
        #B,N,K
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        #B,N
        #距离越小越好，所以有负号
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # print(density.shape)
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6
        # get distance indicator
        #[B,N,N]
        mask = density[:, None, :] > density[:, :, None]
        # print(mask.shape)
        # exit()
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num



class FMS(nn.Module):
    def __init__(self, args):
        super(FMS, self).__init__()
        self.topk_ratio = args['topk_ratio']
        self.cluster_sample_ratio = args['cluster_sample_ratio']

        self.sampler = SortSampler(topk_ratio=self.topk_ratio, input_dim=64, score_pred_net='2layer-fc-256')

    def forward(self, batch_feature):
        L, C, H, W = batch_feature.shape
        dis_priority = torch.ones([L, H, W]).to(batch_feature.device)
        idx = torch.arange(H * W).repeat(L, 1, 1).permute(2, 0, 1).to(batch_feature.device)

        src, sample_reg_loss, sort_confidence_topk, pos_embed = self.sampler(batch_feature.clone(), idx, dis_priority)

        src = src.permute(1, 0, 2)
        _, s_len, _ = src.shape

        cluster_num = max(math.ceil(s_len * self.cluster_sample_ratio), 1)

        idx_cluster, cluster_num = cluster_dpc_knn(src, cluster_num, 10)
        down_dict, idx = merge_tokens(src, idx_cluster, cluster_num, sort_confidence_topk.unsqueeze(2))

        idxxs = []
        for b in range(L):
            i = torch.arange(s_len)
            idxxs.append(idx[b][i])
        idxxs = torch.vstack(idxxs)

        src = index_points(down_dict, idxxs)
        src = src.permute(0, 2, 1)

        pos_embed = pos_embed.permute(1, 2, 0)

        selected_feature_list = []
        spatial_mask_list = []
        for cav_idx in range(L):
            if cav_idx == 0:
                selected_feature = batch_feature[0].flatten(1)
                spatial_mask = torch.ones([H * W, 1], device=src.device)
            else:
                selected_feature = torch.zeros(C, H * W, dtype=src.dtype, device=src.device)
                selected_feature[:, pos_embed[cav_idx][0]] = src[cav_idx]

                spatial_mask = torch.zeros([H * W, 1], device=src.device)
                spatial_mask[pos_embed[cav_idx][0], :] = 1
            selected_feature_list.append(selected_feature)
            spatial_mask_list.append(spatial_mask)

        selected_features = torch.stack(selected_feature_list, 0).reshape(L, C, H, W)
        spatial_masks = torch.stack(spatial_mask_list, 0).reshape(L, H, W, 1)

        return selected_features, spatial_masks
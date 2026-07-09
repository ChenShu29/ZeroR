import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.CoE2E_utils import LayerNorm


def activate_mask_generator(x_updated, topk):
    h = x_updated.size(1)
    w = x_updated.size(2)
    # add all channels
    x_add = torch.sum(x_updated, dim=0)  # [5, 5]
    # reshape
    x_temp = x_add.reshape(-1)  # [25,]
    # initialize mask
    mask = torch.zeros_like(x_temp)  # [25,]
    # sort by descend
    x_sort = torch.sort(x_temp, descending=True)

    mask[x_sort.indices[:topk]] = 1

    mask = mask.reshape(h, w)

    return mask

def spatial_sampling(feat, sample_ratio=0.8):
    N, _, H, W = feat.shape
    sample_pixels = int(H * W * sample_ratio)
    sampled_feat = feat.clone()

    mask_all = []
    for i in range(N):
        # mask = torch.cuda.FloatTensor(H, W).uniform_() > 0.7

        # mask = random_mask_generator(H, W, sample_pixels)
        mask = activate_mask_generator(sampled_feat[i], sample_pixels)
        sampled_feat[i] = sampled_feat[i] * mask

        mask_all.append(mask)

    return sampled_feat, torch.stack(mask_all)


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel, 4, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(4)

        self.conv1_2 = nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        # self.softmax = nn.Softmax(dim=1)

    def min_max_normalize(self, x):
        # Find the minimum and maximum values of the feature map
        min_val = x.min()
        max_val = x.max()
        # Normalize the feature map to have values between 0 and 1
        normalized = (x - min_val) / (max_val - min_val)
        return normalized

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.conv1_2(x_1))

        return self.min_max_normalize(x_1)


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm_a = LayerNorm(dim*2, eps=1e-6, data_format="channels_first")
        self.norm_v = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.a = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, a, v, mask):
        a = self.norm_a(a)
        a = self.a(a)

        v = self.norm_v(v)
        v = self.v(v)

        if mask is not None:
            att = self.proj(a * v * mask)
        else:
            att = self.proj(a * v)

        return att


class COREAttentiveCollaboration(nn.Module):
    def __init__(self, dim=64):
        super(COREAttentiveCollaboration, self).__init__()

        self.pixel_weighted = PixelWeightedFusionSoftmax(dim)
        self.convatt = ConvMod(dim)

    def forward(self, raw_feat):
        """
        Args:
            raw_feat: [num_agent, c, h, w]
        Returns:
            updated_feat: [num_agent, c, h, w]
        """
        num_agent = len(raw_feat)

        local_com_mat = list()
        for i in range(num_agent):
            local_com_mat.append(raw_feat[i, :, :, :])

        local_com_mat_update = local_com_mat.copy()

        conf_map_list = []

        for i in range(num_agent):
            for j in range(num_agent):
                # generate confidence map P and request map R
                conf_map = self.pixel_weighted(local_com_mat[j])
                conf_map_list.append(conf_map)

            ego_request = 1 - conf_map_list[i]

            for k in range(num_agent):
                att_map = ego_request * conf_map_list[k]

                cat_feats = torch.cat([local_com_mat[i], local_com_mat[k]], dim=0)
                q = cat_feats
                v = local_com_mat[k]

                att = self.convatt(q, v, att_map)

                local_com_mat_update[i] = local_com_mat_update[i] + att

        updated_feat = torch.cat(local_com_mat_update, dim=0)

        return updated_feat


class AutoEncoderCORE(nn.Module):
    def __init__(self, feature_num, layer_num, mask_ratio):
        super().__init__()
        self.feature_num = feature_num
        self.feature_stride = 2
        self.mask_ratio = mask_ratio

        self.encoder = nn.ModuleList()

        self.attcoll = COREAttentiveCollaboration(64)

        self.decoder_tsk = nn.ModuleList()
        # self.decoder_rec = nn.ModuleList()

        for i in range(layer_num):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()]

            cur_layers.extend([
                nn.Conv2d(feature_num, feature_num // self.feature_stride,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_num // self.feature_stride,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])

            self.encoder.append(nn.Sequential(*cur_layers))
            feature_num = feature_num // self.feature_stride

        # task decoder
        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [nn.Sequential(
                nn.ConvTranspose2d(feature_num // 2, feature_num, kernel_size=3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]

            cur_layers.extend([nn.Sequential(
                nn.Conv2d(feature_num, feature_num, kernel_size=3,padding=1),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )])
            self.decoder_tsk.append(nn.Sequential(*cur_layers))
            feature_num //= 2

        # # reconstruction decoder
        # feature_num = self.feature_num
        # for i in range(layer_num):
        #     cur_layers = [nn.Sequential(
        #         nn.ConvTranspose2d(
        #             feature_num // 2, feature_num,
        #             kernel_size=2,
        #             stride=2, bias=False
        #         ),
        #         nn.BatchNorm2d(feature_num,
        #                        eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     )]
        #
        #     cur_layers.extend([nn.Sequential(
        #         nn.Conv2d(
        #             feature_num, feature_num, kernel_size=3,
        #             stride=1, bias=False, padding=1
        #         ),
        #         nn.BatchNorm2d(feature_num, eps=1e-3,
        #                        momentum=0.01),
        #         nn.ReLU()
        #     )])
        #     self.decoder_rec.append(nn.Sequential(*cur_layers))
        #     feature_num //= 2

    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        # spatial sampling
        if self.training:
            mask_rate = random.uniform(0.05, 0.55)
        else:
            mask_rate = self.mask_ratio
        x, spatial_masks = spatial_sampling(x, sample_ratio=mask_rate)

        # attentive collaboration
        x = self.attcoll(x)

        # reconstruction
        x_tsk = x
        # x_rec = x
        for i in range(len(self.decoder_tsk)-1, -1, -1):
            x_tsk = self.decoder_tsk[i](x_tsk)
            # x_rec = self.decoder_rec[i](x_rec)

        return x_tsk, spatial_masks
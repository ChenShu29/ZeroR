import torch
import torch.nn as nn

from opencood.models.sub_modules.fax_modules import FAXModule

from opencood.models.sub_modules.resnet_backbone import ResnetEncoder
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.resnet_backbone import NaiveDecoder, CVTDecoder
from opencood.models.sub_modules.base_bev_backbone import BevSegHeadEnhance
from opencood.models.sub_modules.CoE2E_utils import STTF, communication_cal
from opencood.models.fuse_modules.fuse_utils import batch_split, batch_merge
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.sub_modules.torch_transformation_utils import get_roi_and_cav_mask


class BaselineCobevt(nn.Module):
    def __init__(self, args):
        super(BaselineCobevt, self).__init__()

        # Camera Encoder
        self.camera_encoder = ResnetEncoder(args['encoder'])

        self.fax = FAXModule(args['fax'], self.camera_encoder.output_shapes)
        self.mask_para = args['sttf']
        self.sttf = STTF(args['sttf'])

        if 'compress' in args:
            self.compress_flag = True
            self.compress_conv = NaiveCompressor(256, args['compress'])

        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])


        self.semantic_decoder = CVTDecoder(args['cvt_docker'])
        self.sematic_head = BevSegHeadEnhance(args['seg_head_dim'], args['output_class'])
        self.dynamic_head = BevSegHeadEnhance(args['seg_head_dim'], 2)
        self.lane_head = BevSegHeadEnhance(args['seg_head_dim'], 2)

    def forward(self, data_dict):

        camera_data = data_dict['inputs']

        camera_feature = self.camera_encoder(camera_data)

        spatial_features_2d = self.fax(camera_feature, data_dict)

        spatial_features_2d = self.sttf(spatial_features_2d, data_dict['transformation_matrix'], data_dict['record_len'])

        if self.compress_flag:
            spatial_features_2d = self.compress_conv.compress(spatial_features_2d)

        spatial_features_2d = batch_split(spatial_features_2d, data_dict['record_len'])
        transformation_matrix_all = batch_split(batch_merge(data_dict['transformation_matrix'], data_dict['record_len']), data_dict['record_len'])

        fused_bev_feature = []
        communication_volume = []
        communication_rate = []
        for batch in range(len(spatial_features_2d)):
            batch_feature = spatial_features_2d[batch]
            transformation_matrix = transformation_matrix_all[batch]

            if self.compress_flag:
                batch_feature = self.compress_conv.decompress(batch_feature)

            if self.mask_para['use_roi_mask']:
                spatial_mask = get_roi_and_cav_mask(batch_feature.shape, transformation_matrix, self.mask_para['resolution'], self.mask_para['downsample_rate'])
            else:
                spatial_mask = None

            if batch_feature.shape[0] > 1:
                comm_rate, comm_volume = communication_cal(spatial_mask=None, sttf_mask=spatial_mask.permute(3, 0, 1, 2), channels=64)
                communication_volume.append(comm_volume)
                communication_rate.append(comm_rate)

            batch_feature = self.fusion_net(x=batch_feature.unsqueeze(0), mask=spatial_mask.unsqueeze(0))
            batch_feature = batch_feature.squeeze(0)

            fused_bev_feature.append(batch_feature)

        fused_bev_feature = torch.stack(fused_bev_feature, dim=0)
        fused_bev_feature = self.semantic_decoder(fused_bev_feature)

        semantic_logit = self.sematic_head(fused_bev_feature)
        dynamic_logit = self.dynamic_head(fused_bev_feature)
        lane_logit = self.lane_head(fused_bev_feature)

        output_dict = {'semantic_map': semantic_logit,
                       'dynamic_map': dynamic_logit,
                       'lane_map': lane_logit,
                       'communication': {'comm_rates': communication_rate,
                                         'comm_volumes': communication_volume}
                       }

        return output_dict
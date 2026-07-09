import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.sub_modules.resnet_backbone import ResnetEncoder
from opencood.models.sub_modules.resnet_backbone import NaiveDecoder
from opencood.models.sub_modules.base_bev_backbone import BevSegHead


class CorpBEVTFAX(nn.Module):
    def __init__(self, config):
        super(CorpBEVTFAX, self).__init__()
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['fax']
        self.fax = FAXModule(cvm_params, self.encoder.output_shapes)

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(config['target'], config['seg_head_dim'], config['output_class_dynamic'], config['output_class_static'])

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, m, _, _, _ = x.shape

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.fax(x, batch_dict)

        # dynamic head
        x = self.decoder(x)

        # Only for ego
        # x = x[0::2]

        output_dict = self.seg_head(x)

        return output_dict
"""
Post processing for rgb camera groundtruth
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor


opv2v_label_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 0,
    5: 3,
    6: 4,
    7: 4,
    8: 5,
    9: 6,
    10: 4,
    11: 1,
    12: 3,
    13: 0,
    14: 7,
    15: 2,
    16: 0,
    17: 2,
    18: 3,
    19: 0,
    20: 5,
    21: 0,
    22: 7
}

class CameraBevPostprocessor(BasePostprocessor):
    """
    This postprocessor mainly transfer the uint bev maps to float.
    """

    def __init__(self, anchor_params, train):
        super(CameraBevPostprocessor, self).__init__(anchor_params, train)
        self.params = anchor_params
        self.train = train
        self.softmax = nn.Softmax(dim=1)
        self.ignore_index = 0

    def generate_label(self, bev_map):
        """
        Convert rgb images to binary output.

        Parameters
        ----------
        bev_map : np.ndarray
            Uint 8 image with 3 channels.
        """
        bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
        bev_map = np.array(bev_map, dtype=np.float) / 255.

        bev_map[bev_map > 0] = 1

        return bev_map


    def label_remapping(self, ori_map):
        mapping_dict = opv2v_label_mapping
        max_orig = max(mapping_dict.keys())
        lut = np.zeros(max_orig + 1, dtype=np.uint8)

        for orig, new in mapping_dict.items():
            lut[orig] = new

        return lut[ori_map]


    def generate_label_semantic(self, bev_map):
        assert len(np.unique(bev_map)) <= 17, "More than 17 classes occurs"

        bev_map = self.label_remapping(bev_map)
        assert len(np.unique(bev_map)) <= 8, "More than 8 labels occurs"

        labels = np.unique(bev_map)

        scale_factor = 256 / 250
        bev_map = zoom(bev_map, (scale_factor, scale_factor), order=0)

        assert (labels == np.unique(bev_map)).all(), "Label Error Here - Size zoom"

        return bev_map


    def merge_label(self, semantic_map, lane_map, dynamic_map):
        """
        Merge lane and road map into one.

        Parameters
        ----------
        static_map :
        lane_map :
        """
        semantic_map[lane_map == 1] = 8
        semantic_map[dynamic_map == 1] = 9

        return semantic_map

    def generate_occ(self, road_map, dynamic_map):
        merge_map = np.zeros((road_map.shape[0], road_map.shape[1]))
        merge_map[road_map != 1] = 1
        merge_map[dynamic_map == 1] = 1
        return merge_map

    def softmax_argmax(self, seg_logits):
        output_prob = self.softmax(seg_logits)
        output_map = torch.argmax(output_prob, dim=1)

        return output_prob, output_map

    def post_process_train(self, output_dict):
        """
        Post process the output of bev map to segmentation mask.
        todo: currently only for single vehicle bev visualization.

        Parameters
        ----------
        output_dict : dict
            The output dictionary that contains the bev softmax.

        Returns
        -------
        The segmentation map. (B, C, H, W) and (B, H, W)
        """
        static_seg = output_dict['static_seg']
        dynamic_seg = output_dict['dynamic_seg']

        static_prob, static_map = self.softmax_argmax(static_seg)
        dynamic_prob, dynamic_map = self.softmax_argmax(dynamic_seg)

        output_dict.update({
            'static_prob': static_prob,
            'static_map': static_map,
            'dynamic_map': dynamic_map,
            'dynamic_prob': dynamic_prob
        })

        return output_dict

    def semantic_post_process(self, output_dict, batch_dict, index_lane=8, index_dynamic=9):
        semantic_logit = output_dict['semantic_map'].detach()
        semantic_map = torch.argmax(semantic_logit, dim=1)

        dynamic_logit = output_dict['dynamic_map'].detach()
        dynamic_map = torch.argmax(dynamic_logit, dim=1)

        lane_logit = output_dict['lane_map'].detach()
        lane_map = torch.argmax(lane_logit, dim=1)

        semantic_gt = batch_dict['gt_semantic'].detach()
        dynamic_gt = batch_dict['gt_dynamic'].detach()
        lane_gt = batch_dict['gt_lane'].detach()

        # merge the whole semantic_map
        complete_map = semantic_map.clone()
        complete_map[lane_map == 1] = index_lane
        complete_map[dynamic_map == 1] = index_dynamic

        complete_gt = semantic_gt.clone()
        complete_gt[lane_gt == 1] = index_lane
        complete_gt[dynamic_gt == 1] = index_dynamic

        output_dict.update({
            'semantic_map': semantic_map,
            'dynamic_map': dynamic_map,
            'lane_map': lane_map,
            'semantic_gt': semantic_gt,
            'dynamic_gt': dynamic_gt,
            'lane_gt': lane_gt,
            'complete_map': complete_map,   # Only for Visualization
            'complete_gt': complete_gt,
        })

        return output_dict

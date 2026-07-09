"""
Late fusion for camera.
"""

import random
from collections import OrderedDict
import math

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets import base_camera_dataset
from opencood.utils import common_utils


class CamLateFusionDataset(base_camera_dataset.BaseCameraDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamLateFusionDataset, self).__init__(params, visualize, train,
                                                   validate)
        self.visible = params['train_params']['visible']

    def __getitem__(self, idx):
        data_sample = self.get_sample(idx)
        if self.train:
            return self.get_item_train(data_sample)
        else:
            return self.get_item_test(data_sample)

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        selected_cav_id, selected_cav_base = random.choice(list(base_data_dict.items()))

        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        selected_cav_processed = self.get_single_cav(selected_cav_base)
        selected_cav_processed.update({'transformation_matrix': transformation_matrix})

        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert cav_id == list(base_data_dict.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2
                                 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            # find the transformation matrix from current cav to ego.
            # this is used to project prediction to the right space
            transformation_matrix = selected_cav_base['params']['transformation_matrix']
            selected_cav_processed = self.get_single_cav(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix': transformation_matrix})

            processed_data_dict.update({cav_id: selected_cav_processed})

        return processed_data_dict

    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for late fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict({'camera': OrderedDict()})

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            camera_data = self.pre_processor.preprocess(camera_data)

            camera_intrinsic = selected_cav_base['camera_params'][camera_id]['camera_intrinsic']
            cam2ego = selected_cav_base['camera_params'][camera_id]['camera_extrinsic_to_ego']

            camera_dict = {
                'data': camera_data,
                'intrinsic': camera_intrinsic,
                'extrinsic': cam2ego
            }
            selected_cav_processed['camera'].update({camera_id: camera_dict})

        # process the groundtruth
        semantic_bev = self.post_processor.generate_label_semantic(selected_cav_base['semantic_2d.npy'])
        lane_bev = self.post_processor.generate_label(selected_cav_base['bev_lane.png'])
        if self.visible and self.train:
            dynamic_bev = self.post_processor.generate_label(selected_cav_base['bev_visibility.png'])
        else:
            dynamic_bev = self.post_processor.generate_label(selected_cav_base['bev_visibility_corp.png'])
        # semantic_bev = self.post_processor.merge_label(semantic_bev, lane_bev, dynamic_bev)

        gt_dict = {'semantic_bev': semantic_bev,
                   'lane_bev': lane_bev,
                   'dynamic_bev': dynamic_bev}

        selected_cav_processed.update({'gt': gt_dict})

        return selected_cav_processed

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        if not self.train:
            assert len(batch) == 1

        output_dict = {'ego': {}}

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cam_intrinsic_all_batch = []

        gt_semantic_all_batch = []
        gt_dynamic_all_batch = []
        gt_lane_all_batch = []

        transformation_matrix_all_batch = []

        # loop all scenes
        for i in range(len(batch)):
            cur_scene_data = batch[i]

            cam_rgb_all_agents = []
            cam_to_ego_all_agents = []
            cam_intrinsic_all_agents = []

            gt_semantic_all_agents = []
            gt_dynamic_all_agents = []
            gt_lane_all_agents = []
            transformation_matrix_all_agents = []

            # loop all agents
            for agent_id, _ in cur_scene_data.items():
                camera_data = cur_scene_data[agent_id]['camera']

                cam_rgb_cur_agent = []
                cam_to_ego_cur_agent = []
                cam_intrinsic_cur_agent = []

                # loop all cameras
                for camera_id, camera_content in camera_data.items():
                    cam_rgb_cur_agent.append(camera_content['data'])
                    cam_to_ego_cur_agent.append(camera_content['extrinsic'])
                    cam_intrinsic_cur_agent.append(camera_content['intrinsic'])

                # M, H, W, 3 -> M is the num of cameras
                cam_rgb_cur_agent = np.stack(cam_rgb_cur_agent)
                cam_to_ego_cur_agent = np.stack(cam_to_ego_cur_agent)
                cam_intrinsic_cur_agent = np.stack(cam_intrinsic_cur_agent)

                cam_rgb_all_agents.append(cam_rgb_cur_agent)
                cam_to_ego_all_agents.append(cam_to_ego_cur_agent)
                cam_intrinsic_all_agents.append(cam_intrinsic_cur_agent)

                # append groundtruth, H,W
                semantic_bev = cur_scene_data[agent_id]['gt']['semantic_bev']
                gt_semantic_all_agents.append(semantic_bev)
                gt_dynamic_all_agents.append(cur_scene_data[agent_id]['gt']['dynamic_bev'])
                gt_lane_all_agents.append(cur_scene_data[agent_id]['gt']['lane_bev'])

                transformation_matrix = cur_scene_data[agent_id]['transformation_matrix']
                transformation_matrix_all_agents.append(transformation_matrix)

            # gather all data from different batches together,
            # (L,M,H,W,3) -> L is the num of agents
            cam_rgb_all_agents = np.stack(cam_rgb_all_agents)
            cam_to_ego_all_agents = np.stack(cam_to_ego_all_agents)
            cam_intrinsic_all_agents = np.stack(cam_intrinsic_all_agents)

            # (L, H, W)
            gt_semantic_all_agents = np.stack(gt_semantic_all_agents)
            gt_dynamic_all_agents = np.stack(gt_dynamic_all_agents)
            gt_lane_all_agents = np.stack(gt_lane_all_agents)

            # (L, 4, 4)
            transformation_matrix_all_agents = np.stack(transformation_matrix_all_agents)

            # Append to batches
            cam_rgb_all_batch.append(cam_rgb_all_agents)
            cam_to_ego_all_batch.append(cam_to_ego_all_agents)
            cam_intrinsic_all_batch.append(cam_intrinsic_all_agents)
            gt_semantic_all_batch.append(gt_semantic_all_agents)
            gt_dynamic_all_batch.append(gt_dynamic_all_agents)
            gt_lane_all_batch.append(gt_lane_all_agents)
            transformation_matrix_all_batch.append(transformation_matrix_all_agents)

        # groundtruth gather (B,L,H,W)
        gt_semantic_all_batch = torch.from_numpy(np.stack(gt_semantic_all_batch)).long()
        gt_dynamic_all_batch = torch.from_numpy(np.stack(gt_dynamic_all_batch)).long()
        gt_lane_all_batch = torch.from_numpy(np.stack(gt_lane_all_batch)).long()

        # input data gather (B,L,M,H,W,C)
        cam_rgb_all_batch = torch.from_numpy(np.concatenate(cam_rgb_all_batch, axis=0)).float()
        cam_intrinsic_all_batch = torch.from_numpy(np.concatenate(cam_intrinsic_all_batch, axis=0)).float()
        cam_to_ego_all_batch = torch.from_numpy(np.concatenate(cam_to_ego_all_batch, axis=0)).float()

        # (B,L,4,4)
        transformation_matrix_all_batch = torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()

        # convert numpy arrays to torch tensor
        output_dict['ego'].update({
            'inputs': cam_rgb_all_batch,
            'extrinsic': cam_to_ego_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'gt_semantic': gt_semantic_all_batch,
            'gt_dynamic': gt_dynamic_all_batch,
            'gt_lane': gt_lane_all_batch,
            'transformation_matrix': transformation_matrix_all_batch
        })

        return output_dict

    def post_process(self, output_dict, batch_dict):
        output_dict = self.post_processor.semantic_post_process(output_dict, batch_dict)

        return output_dict

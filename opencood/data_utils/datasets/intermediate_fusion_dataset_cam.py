"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch
import math

import opencood
from opencood.data_utils.datasets import base_camera_dataset
from opencood.utils import common_utils


class CamIntermediateFusionDataset(base_camera_dataset.BaseCameraDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamIntermediateFusionDataset, self).__init__(params, visualize, train, validate)
        self.visible = params['train_params']['visible']

    def __getitem__(self, idx):
        data_sample = self.get_sample(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = OrderedDict()

        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in data_sample.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(data_sample.keys())[0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = self.get_pairwise_transformation(data_sample, self.params['train_params']['max_cav'])

        # Final shape: (L, M, H, W, 3)
        camera_data = []
        # (L, M, 3, 3)
        camera_intrinsic = []
        # (L, M, 4, 4)
        camera2ego = []

        # (max_cav, 4, 4)
        transformation_matrix = []
        # (1, H, W)
        gt_lane = []
        gt_dynamic = []
        # (1, h, w)
        gt_semantic = []

        # temporal data load
        if self.temporal_params['temporal']:
            previous_camera_data = []
            previous_camera_intrinsic = []
            previous_camera_extrinsic = []
            previous_transformation_matrix = []
            future_semantic, future_lane, future_dynamic = [], [], []

        # loop over all CAVs to process information, and save information from all agents to 'ego' data platform
        for cav_id, selected_cav_base in data_sample.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2
                                 + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                print("Debug Flag: Remove Vehicle Far Away")  # No print, since it has been removed in self.get_sample(idx)
                continue

            selected_cav_processed = self.get_single_cav(selected_cav_base)

            camera_data.append(selected_cav_processed['camera']['data'])
            camera_intrinsic.append(selected_cav_processed['camera']['intrinsic'])
            camera2ego.append(selected_cav_processed['camera']['extrinsic'])
            transformation_matrix.append(selected_cav_processed['transformation_matrix'])

            if self.temporal_params['temporal']:
                previous_camera_data.append(selected_cav_processed['previous_camera'])
                previous_camera_intrinsic.append(selected_cav_processed['previous_camera_intrinsic'])
                previous_camera_extrinsic.append(selected_cav_processed['previous_camera_extrinsic'])
                previous_transformation_matrix.append(selected_cav_processed['previous_transformation_matrix'])

            if cav_id == ego_id:
                gt_semantic.append(selected_cav_processed['gt']['semantic_bev'])
                gt_lane.append(selected_cav_processed['gt']['lane_bev'])
                gt_dynamic.append(selected_cav_processed['gt']['dynamic_bev'])

                if self.temporal_params['temporal']:
                    future_semantic.append(selected_cav_processed['gt']['future_semantic'])
                    future_lane.append(selected_cav_processed['gt']['future_lane'])
                    future_dynamic.append(selected_cav_processed['gt']['future_dynamic'])

        # stack all agents together
        camera_data = np.stack(camera_data)
        camera_intrinsic = np.stack(camera_intrinsic)
        camera2ego = np.stack(camera2ego)

        gt_semantic = gt_semantic[0]
        gt_lane = gt_lane[0]
        gt_dynamic = gt_dynamic[0]

        transformation_matrix = np.stack(transformation_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(transformation_matrix), 1, 1))
        transformation_matrix = np.concatenate([transformation_matrix, padding_eye], axis=0)


        # camera_extrinsic here is camera to ego, not camera to lidar
        if self.temporal_params['temporal']:
            processed_data_dict['ego'].update({
                'transformation_matrix': transformation_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'camera_data': camera_data,
                'previous_camera_data': np.stack(previous_camera_data, axis=0),
                'previous_camera_intrinsic': np.stack(previous_camera_intrinsic, axis=0),
                'previous_camera_extrinsic': np.stack(previous_camera_extrinsic, axis=0),
                'previous_transformation_matrix': np.stack(previous_transformation_matrix, axis=0),
                'camera_intrinsic': camera_intrinsic,
                'camera_extrinsic': camera2ego,
                'gt_semantic': gt_semantic,
                'gt_dynamic': gt_dynamic,
                'gt_lane': gt_lane,
                'gt_future_semantic': future_semantic[0],
                'gt_future_lane': future_lane[0],
                'gt_future_dynamic': future_dynamic[0]})
        else:
            processed_data_dict['ego'].update({
                'transformation_matrix': transformation_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'camera_data': camera_data,
                'camera_intrinsic': camera_intrinsic,
                'camera_extrinsic': camera2ego,
                'gt_semantic': gt_semantic,
                'gt_lane': gt_lane,
                'gt_dynamic': gt_dynamic,
                'timestamp_path': data_sample[cav_id]['params']['timestamp_path']})

        return processed_data_dict    # -> next step is collect_batch

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()

        # update the transformation matrix
        transformation_matrix = selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({'transformation_matrix': transformation_matrix})

        # for intermediate fusion, we only need ego's gt
        if selected_cav_base['ego']:
            # process the groundtruth
            semantic_bev = self.post_processor.generate_label_semantic(selected_cav_base['semantic_2d.npy'])
            lane_bev = self.post_processor.generate_label(selected_cav_base['bev_lane.png'])
            dynamic_bev = self.post_processor.generate_label(selected_cav_base['bev_visibility_corp.png'])

            if self.temporal_params['temporal']:
                future_dynamic_list = []
                future_lane_list = []
                future_semantic_list = []
                for step in selected_cav_base['future_dynamic'].keys():
                    future_dynamic_list.append(self.post_processor.generate_label(selected_cav_base['future_dynamic'][step]))
                    future_lane_list.append(self.post_processor.generate_label(selected_cav_base['future_lane'][step]))
                    future_semantic_list.append(self.post_processor.generate_label_semantic(selected_cav_base['future_semantic'][step]))

                gt_dict = {'semantic_bev': semantic_bev,
                           'lane_bev': lane_bev,
                           'dynamic_bev': dynamic_bev,
                           'future_semantic': np.stack(future_semantic_list),
                           'future_lane': np.stack(future_lane_list),
                           'future_dynamic': np.stack(future_dynamic_list),}
            else:
                gt_dict = {'semantic_bev': semantic_bev,
                           'lane_bev': lane_bev,
                           'dynamic_bev': dynamic_bev}

            selected_cav_processed.update({'gt': gt_dict})

        all_camera_data = []
        # all_camera_origin = []
        all_camera_intrinsic = []
        all_camera_extrinsic = []

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            # all_camera_origin.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)
            camera_intrinsic = selected_cav_base['camera_params'][camera_id]['camera_intrinsic']
            cam2ego = selected_cav_base['camera_params'][camera_id]['camera_extrinsic_to_ego']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)

        camera_dict = {
            # 'origin_data': np.stack(all_camera_origin),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic)
        }

        selected_cav_processed.update({'camera': camera_dict})

        # preprocess input rgb image in previous timestamp
        if self.temporal_params['temporal']:
            all_previous_camera_data = []
            all_previous_camera_intrinsic = []
            all_previous_camera_extrinsic = []
            previous_transformation_matrix = []
            for step, camera_content in selected_cav_base['camera_np_previous'].items():
                previous_camera_data = []
                previous_camera_intrinsic = []
                previous_camera_extrinsic = []
                for camera_id, camera_data in camera_content.items():
                    previous_camera_data.append(self.pre_processor.preprocess(camera_data))
                    previous_camera_intrinsic.append(selected_cav_base['camera_params_previous'][step][camera_id]['camera_intrinsic'])
                    previous_camera_extrinsic.append(selected_cav_base['camera_params_previous'][step][camera_id]['camera_extrinsic_to_ego'])

                all_previous_camera_data.append(np.stack(previous_camera_data))
                all_previous_camera_intrinsic.append(np.stack(previous_camera_intrinsic))
                all_previous_camera_extrinsic.append(np.stack(previous_camera_extrinsic))

                previous_transformation_matrix.append(selected_cav_base['params_previous'][step]['transformation_matrix'])

            selected_cav_processed.update({'previous_camera': np.stack(all_previous_camera_data)})
            selected_cav_processed.update({'previous_transformation_matrix': np.stack(previous_transformation_matrix)})
            selected_cav_processed.update({'previous_camera_intrinsic': np.stack(all_previous_camera_intrinsic)})
            selected_cav_processed.update({'previous_camera_extrinsic': np.stack(all_previous_camera_extrinsic)})

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
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []
        timestamp_paths = []

        if self.temporal_params['temporal']:
            previous_cam_rgb_all_batch = []
            previous_cam_intrinsic_all_batch = []
            previous_cam_extrinsic_all_batch = []
            previous_trans_matrix_all_batch = []
            future_gt_semantic_all_batch = []
            future_gt_lane_all_batch = []
            future_gt_dynamic_all_batch = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']

            # [num_CAV, V, H, W, 3]
            camera_data = ego_dict['camera_data']
            camera_intrinsic = ego_dict['camera_intrinsic']
            camera_extrinsic = ego_dict['camera_extrinsic']

            assert camera_data.shape[0] == camera_intrinsic.shape[0] == camera_extrinsic.shape[0]

            record_len.append(camera_data.shape[0])

            cam_rgb_all_batch.append(camera_data)
            cam_intrinsic_all_batch.append(camera_intrinsic)
            cam_to_ego_all_batch.append(camera_extrinsic)

            # ground truth
            gt_semantic_all_batch.append(ego_dict['gt_semantic'])
            gt_dynamic_all_batch.append(ego_dict['gt_dynamic'])
            gt_lane_all_batch.append(ego_dict['gt_lane'])

            # transformation matrix
            transformation_matrix_all_batch.append(ego_dict['transformation_matrix'])
            # pairwise matrix
            pairwise_t_matrix_all_batch.append(ego_dict['pairwise_t_matrix'])

            timestamp_paths.append(ego_dict['timestamp_path'])


            if self.temporal_params['temporal']:

                previous_cam_rgb_all_batch.append(ego_dict['previous_camera_data'])
                previous_cam_intrinsic_all_batch.append(ego_dict['previous_camera_intrinsic'])
                previous_cam_extrinsic_all_batch.append(ego_dict['previous_camera_extrinsic'])
                previous_trans_matrix_all_batch.append(ego_dict['previous_transformation_matrix'])

                future_gt_semantic_all_batch.append(ego_dict['gt_future_semantic'])
                future_gt_lane_all_batch.append(ego_dict['gt_future_lane'])
                future_gt_dynamic_all_batch.append(ego_dict['gt_future_dynamic'])

        # (B*L, 1, M, H, W, C)
        cam_rgb_all_batch = torch.from_numpy(np.concatenate(cam_rgb_all_batch, axis=0)).float()
        cam_intrinsic_all_batch = torch.from_numpy(np.concatenate(cam_intrinsic_all_batch, axis=0)).float()
        cam_to_ego_all_batch = torch.from_numpy(np.concatenate(cam_to_ego_all_batch, axis=0)).float()

        # (B,)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # (B, 1, H, W)
        gt_semantic_all_batch = torch.from_numpy(np.stack(gt_semantic_all_batch)).long()
        gt_dynamic_all_batch = torch.from_numpy(np.stack(gt_dynamic_all_batch)).long()
        gt_lane_all_batch = torch.from_numpy(np.stack(gt_lane_all_batch)).long()

        # (B,max_cav,4,4)
        transformation_matrix_all_batch = torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()
        pairwise_t_matrix_all_batch = torch.from_numpy(np.stack(pairwise_t_matrix_all_batch)).float()

        if self.temporal_params['temporal']:
            previous_cam_rgb_all_batch = torch.from_numpy(np.concatenate(previous_cam_rgb_all_batch, axis=0)).float()
            previous_cam_intrinsic_all_batch = torch.from_numpy(np.concatenate(previous_cam_intrinsic_all_batch, axis=0)).float()
            previous_cam_extrinsic_all_batch = torch.from_numpy(np.concatenate(previous_cam_extrinsic_all_batch, axis=0)).float()
            previous_trans_matrix_all_batch = torch.from_numpy(np.concatenate(previous_trans_matrix_all_batch, axis=0)).float()

            future_gt_semantic_all_batch = torch.from_numpy(np.stack(future_gt_semantic_all_batch)).long()
            future_gt_lane_all_batch = torch.from_numpy(np.stack(future_gt_lane_all_batch)).long()
            future_gt_dynamic_all_batch = torch.from_numpy(np.stack(future_gt_dynamic_all_batch)).long()

            output_dict['ego'].update({
                'inputs': cam_rgb_all_batch,                                         
                'extrinsic': cam_to_ego_all_batch,                                  
                'intrinsic': cam_intrinsic_all_batch,                               
                'gt_semantic': future_gt_semantic_all_batch,                         
                'gt_dynamic': future_gt_dynamic_all_batch,                         
                'gt_lane': future_gt_lane_all_batch,                               
                'transformation_matrix': transformation_matrix_all_batch,           
                # 'pairwise_t_matrix': pairwise_t_matrix_all_batch,                
                'record_len': record_len,                                           
                'previous_camera': previous_cam_rgb_all_batch,                      
                'previous_camera_intrinsic': previous_cam_intrinsic_all_batch,     
                'previous_camera_extrinsic': previous_cam_extrinsic_all_batch,     
                'previous_transformation_matrix': previous_trans_matrix_all_batch,  
            })

        else:
            output_dict['ego'].update({
                'inputs': cam_rgb_all_batch,                             
                'extrinsic': cam_to_ego_all_batch,                         
                'intrinsic': cam_intrinsic_all_batch,                   
                'gt_semantic': gt_semantic_all_batch,                
                'gt_dynamic': gt_dynamic_all_batch,
                'gt_lane': gt_lane_all_batch,
                'transformation_matrix': transformation_matrix_all_batch,   
                'pairwise_t_matrix': pairwise_t_matrix_all_batch,      
                'record_len': record_len,
                'timestamp_path': timestamp_paths
            })

        return output_dict

    def post_process(self, output_dict, batch_dict):
        output_dict = self.post_processor.semantic_post_process(output_dict, batch_dict)

        return output_dict

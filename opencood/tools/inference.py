# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>, Yijie Chen
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import warnings
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_semantic

warnings.filterwarnings(
    "ignore",
    message=".*torch.meshgrid.*indexing argument.*",
    category=UserWarning,
)


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--target_type', type=str,
                        help='Target segmentation type')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result in npy_test file')
    parser.add_argument('--cal_comm', action='store_true',
                        help='whether to calculate communication metrics')

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    print('------------------Dataset Building------------------')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, validate=False)   # Dataset validate_dir
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=12,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('------------------Creating   Model------------------')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('------------Loading Model from checkpoint-----------')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # Evaluation Matrices  -  Segmentation Task
    iou_all = []
    mean_iou = []
    if opt.cal_comm:
        comm_rates = []
        comm_volumes = []


    for i, batch_data in tqdm(enumerate(data_loader)):
        with (torch.no_grad()):
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            output_dict = model(batch_data['ego'])

            if opt.cal_comm:
                comm_rates.append(output_dict['communication']['comm_rates'])
                comm_volumes.append(output_dict['communication']['comm_volumes'])

            # Calculate the IoU results
            output_dict_iou = opencood_dataset.post_process(output_dict, batch_data['ego'])
            iou, m_iou = cal_iou_semantic(output_dict_iou)

            iou_all.append(iou)
            mean_iou.append(m_iou)

    if opt.cal_comm:
        eval_utils.eval_final_results_seg(iou_all=iou_all, mean_iou_all=mean_iou, comm_rates=comm_rates, comm_volume=comm_volumes)
    else:
        eval_utils.eval_final_results_seg(iou_all=iou_all, mean_iou_all=mean_iou)


if __name__ == '__main__':
    main()

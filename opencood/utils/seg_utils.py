import math
import torch
import torch.nn.functional as F

import numpy as np



def mean_precision(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    mAP = [0] * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)
        val = n_ii / float(n_ij)
        if math.isnan(val):
            mAP[i] = 0.
        else:
            mAP[i] = val
    # print(mAP)
    return mAP


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    return IU


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def cal_iou_training(output_dict, batch_dict):
    """
    Calculate IoU during training.

    Parameters
    ----------
    output_dict : dict
        The output directory with predictions.

    batch_dict: dict
        The data that contains the gt.

    Returns
    -------
    The iou for static and dynamic bev map.
    """

    batch_size = batch_dict['ego']['gt_static'].shape[0]

    for i in range(batch_size):

        # [B*l, 1, H, W]  ->  [H, W]   (l==1: ego)
        gt_static = batch_dict['ego']['gt_static'].detach().cpu().data.numpy()[i, 0]
        gt_static = np.array(gt_static, dtype=np.int)

        gt_dynamic = batch_dict['ego']['gt_dynamic'].detach().cpu().data.numpy()[i, 0]
        gt_dynamic = np.array(gt_dynamic, dtype=np.int)

        # [B*l, H, W]   ->  [H, W]    (l==1, because fused to ego)   (base 暂时适用，因为 test batch size = 1，所以即便没 fusion，只取第一个就是 ego)
        pred_static = output_dict['static_map'].detach().cpu().data.numpy()[i]
        pred_static = np.array(pred_static, dtype=np.int)

        pred_dynamic = output_dict['dynamic_map'].detach().cpu().data.numpy()[i]
        pred_dynamic = np.array(pred_dynamic, dtype=np.int)

        iou_dynamic = mean_IU(pred_dynamic, gt_dynamic)
        iou_static = mean_IU(pred_static, gt_static)

        return iou_dynamic, iou_static


def cal_iou_semantic(output_dict, n_classes=8):
        # semantic_map = output_dict['semantic_map']
        semantic_map_labeled = output_dict['semantic_map'].clone()
        dynamic_map = output_dict['dynamic_map']
        lane_map = output_dict['lane_map']

        # TODO: 将 GT 中没有标签的点，在分割图中也打成 unlabeled, 这样不会影响并集

        semantic_gt = output_dict['semantic_gt']
        dynamic_gt = output_dict['dynamic_gt']
        lane_gt = output_dict['lane_gt']

        semantic_map_labeled[semantic_gt==0] = 0

        # Compute IoU for all Semantic
        semantic_pred_oh = F.one_hot(semantic_map_labeled, n_classes).permute(0, 3, 1, 2)
        semantic_gt_oh = F.one_hot(semantic_gt, n_classes).permute(0, 3, 1, 2)
        semantic_intersection = (semantic_pred_oh & semantic_gt_oh).sum(dim=(2, 3))
        semantic_union = (semantic_pred_oh | semantic_gt_oh).sum(dim=(2, 3))
        iou = semantic_intersection.sum(0) / (semantic_union.sum(0) + 1e-10)


        # Compute IoU for Lanes
        lane_pred_oh = F.one_hot(lane_map, 2).permute(0, 3, 1, 2)
        lane_gt_oh = F.one_hot(lane_gt, 2).permute(0, 3, 1, 2)
        lane_intersection = (lane_pred_oh & lane_gt_oh).sum(dim=(2, 3))
        lane_union = (lane_pred_oh | lane_gt_oh).sum(dim=(2, 3))
        iou_lane = (lane_intersection.sum(0) / (lane_union.sum(0) + 1e-10))[1]
        iou = torch.cat((iou, iou_lane.unsqueeze(0)), dim=0)


        # Compute IoU for Dynamics
        dynamic_pred_oh = F.one_hot(dynamic_map, 2).permute(0, 3, 1, 2)
        dynamic_gt_oh = F.one_hot(dynamic_gt, 2).permute(0, 3, 1, 2)
        dynamic_intersection = (dynamic_pred_oh & dynamic_gt_oh).sum(dim=(2, 3))
        dynamic_union = (dynamic_pred_oh | dynamic_gt_oh).sum(dim=(2, 3))
        iou_dynamic = (dynamic_intersection.sum(0) / (dynamic_union.sum(0) + 1e-10))[1]
        iou = torch.cat((iou, iou_dynamic.unsqueeze(0)), dim=0)

        assert iou.shape[0] == 10, "Iou items errors"

        m_iou = iou[1:].mean().item()

        return iou.cpu().data.numpy(), m_iou


def cal_iou_semantic_test(output_dict, n_classes=18):
    semantic_map = output_dict['semantic_map']
    target_map = output_dict['target_map']

    iou, m_iou = [], 0

    for i in range(n_classes):
        intersection = np.logical_and(target_map == i, semantic_map == i)
        union = np.logical_or(target_map == i, semantic_map == i)
        temp = np.sum(intersection) / np.sum(union)
        iou.append(temp)
        m_iou += temp
    m_iou = m_iou / n_classes

    return iou, m_iou


def cal_iou_semantic_multi(output_dict,n_classes=8):
    semantic_map_labeled = output_dict['semantic_map'].clone()
    dynamic_map = output_dict['dynamic_map']
    lane_map = output_dict['lane_map']

    # TODO: 将 GT 中没有标签的点，在分割图中也打成 unlabeled, 这样不会影响并集

    semantic_gt = output_dict['semantic_gt']
    dynamic_gt = output_dict['dynamic_gt']
    lane_gt = output_dict['lane_gt']

    semantic_map_labeled[semantic_gt == 0] = 0

    # Compute IoU for all Semantic
    semantic_pred_oh = F.one_hot(semantic_map_labeled, n_classes).permute(0, 3, 1, 2)
    semantic_gt_oh = F.one_hot(semantic_gt, n_classes).permute(0, 3, 1, 2)
    semantic_intersection = (semantic_pred_oh & semantic_gt_oh).sum(dim=(2, 3))
    semantic_union = (semantic_pred_oh | semantic_gt_oh).sum(dim=(2, 3))
    iou = semantic_intersection / (semantic_union + 1e-10)

    # Compute IoU for Lanes
    lane_pred_oh = F.one_hot(lane_map, 2).permute(0, 3, 1, 2)
    lane_gt_oh = F.one_hot(lane_gt, 2).permute(0, 3, 1, 2)
    lane_intersection = (lane_pred_oh & lane_gt_oh).sum(dim=(2, 3))
    lane_union = (lane_pred_oh | lane_gt_oh).sum(dim=(2, 3))
    iou_lane = (lane_intersection / (lane_union + 1e-10))[:,1:2]
    iou = torch.cat((iou, iou_lane), dim=1)

    # Compute IoU for Dynamics
    dynamic_pred_oh = F.one_hot(dynamic_map, 2).permute(0, 3, 1, 2)
    dynamic_gt_oh = F.one_hot(dynamic_gt, 2).permute(0, 3, 1, 2)
    dynamic_intersection = (dynamic_pred_oh & dynamic_gt_oh).sum(dim=(2, 3))
    dynamic_union = (dynamic_pred_oh | dynamic_gt_oh).sum(dim=(2, 3))
    iou_dynamic = (dynamic_intersection / (dynamic_union + 1e-10))[:,1:2]
    iou = torch.cat((iou, iou_dynamic), dim=1)

    # [PS, 10]
    assert iou.shape[1] == 10, "Iou items errors"
    # [PS]
    m_iou = torch.mean(iou[:, 1:], dim=1)

    return iou.cpu().data.numpy(), m_iou.cpu().data.numpy()





class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

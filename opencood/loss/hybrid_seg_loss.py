import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.loss.lovasz_loss import Lovasz_softmax


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1.0, ignore_label=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label

    def forward(self, predict, target, class_weight):
        ce_loss = F.cross_entropy(predict, target, weight=class_weight, ignore_index=self.ignore_label, reduction='none')
        pt = torch.exp(-ce_loss.detach())
        pt = pt.clamp(min=1e-5, max=1 - 1e-5)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        mask = target != self.ignore_label
        return loss.sum() / mask.sum() * 10


class FocalSoftmaxLoss(nn.Module):
    def __init__(self, ignore_label, gamma=1, softmax=True):
        super(FocalSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.ignore_label = ignore_label

        self.softmax = softmax

    def forward(self, x, target, alpha):
        '''compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        '''

        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x

        target = target.view(-1, 1)
        mask = target != self.ignore_label

        if self.softmax:
            pred_softmax = F.softmax(pred, 1)
        else:
            pred_softmax = pred
        pred_softmax = pred_softmax.gather(1, target).view(-1)
        pred_logsoft = pred_softmax.clamp(1e-6).log()
        alpha = alpha.gather(0, target.squeeze())
        loss = - (1-pred_softmax).pow(self.gamma)
        loss = loss * pred_logsoft * alpha
        if mask is not None:
            if len(mask.size()) > 1:
                mask = mask.view(-1)
            loss = (loss * mask).sum() / mask.sum()
            return loss
        else:
            return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_idx=None, weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_idx = ignore_idx
        self.class_weights = weight

    def forward(self, pred, target):
        valid_mask = torch.ones(pred.shape[1], dtype=torch.bool, device=pred.device)
        valid_mask[self.ignore_idx] = False

        target_valid = target.clone()
        for idx in self.ignore_idx:
            target_valid[target_valid == idx] = -1

        pred = F.softmax(pred, dim=1) * valid_mask.view(1,-1,1,1)
        target_valid = F.one_hot(torch.clamp(target_valid, min=0), num_classes=pred.shape[1]).permute(0,3,1,2).float()
        target_valid = target_valid * valid_mask.view(1, -1, 1, 1)

        intersection = (pred * target_valid).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_valid.sum(dim=(2,3))

        dice = (2.*intersection[:, valid_mask] + self.smooth)/(union[:, valid_mask] + self.smooth)
        weighted_loss = self.class_weights[valid_mask] * (1 - dice)
        dice_loss = weighted_loss.sum(dim=1) / weighted_loss.shape[1]

        return dice_loss.mean()

class OriDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weight=None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = weight

    def forward(self, pred, target):

        pred = F.softmax(pred, dim=1)
        target_valid = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()

        intersection = (pred * target_valid).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_valid.sum(dim=(2,3))

        dice = (2.*intersection + self.smooth)/(union + self.smooth)
        weighted_loss = self.class_weights * (1 - dice)
        dice_loss = weighted_loss.sum(dim=1)
        return dice_loss.mean()



class HybridSegLoss(nn.Module):
    def __init__(self, args):
        super(HybridSegLoss, self).__init__()

        # self.coe = args['coe']
        self.weights = args['weights']
        # self.weights = nn.Parameter(torch.ones(3).cuda())
        # class_frequency = torch.from_numpy(np.array(args['class_frequency']).astype(np.float32))
        # self.class_weight = nn.Parameter(100. / class_frequency).cuda()
        self.class_weight = torch.from_numpy(np.array(args['class_weight']).astype(np.float32)).cuda()
        self.exclude_idx = args['exclude_idx']
        self.ignore_label = args['exclude_idx'][0]

        # self.semantic_focal_loss = FocalLoss(alpha=args['alpha'], gamma=args['gamma'], ignore_label=self.ignore_label)
        self.semantic_focal_loss = FocalSoftmaxLoss(ignore_label=self.ignore_label, gamma=2, softmax=True)
        self.semantic_lovasz_loss = Lovasz_softmax(ignore=self.ignore_label)
        self.semantic_dice_loss = DiceLoss(smooth=1.0, ignore_idx=self.exclude_idx, weight=self.class_weight.cuda())
        # self.semantic_ce_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_label, reduction='mean')

        self.loss_func_dynamic = nn.CrossEntropyLoss(weight=torch.Tensor([1., 50]).cuda())
        # self.dynamic_dice_loss = OriDiceLoss(smooth=1.0, weight=torch.Tensor([1., 5.]).cuda())

        self.loss_func_static = nn.CrossEntropyLoss(weight=torch.Tensor([1., 50]).cuda())
        # self.static_dice_loss = OriDiceLoss(smooth=1.0, weight=torch.Tensor([1., 3.5]).cuda())

        self.loss_dict = {}

    def forward(self, output_dict, gt_dict, epoch, epoch_num):

        semantic_pred = output_dict['semantic_map']
        semantic_gt = gt_dict['gt_semantic'].squeeze(1)
        # semantic_gt_ignored = torch.where(
        #     torch.isin(semantic_gt, torch.tensor(self.exclude_idx, device=semantic_gt.device)), self.ignore_label, semantic_gt)

        dynamic_pred = output_dict['dynamic_map']
        dynamic_gt = gt_dict['gt_dynamic'].squeeze(1)

        lane_pred = output_dict['lane_map']
        lane_gt = gt_dict['gt_lane'].squeeze(1)

        if epoch < epoch_num / 3:
            focal_weight, dice_weight, lovasz_weight = 0.5, 0.2, 0.3
        elif epoch_num / 3 <= epoch < epoch_num / 3 * 2:
            focal_weight, dice_weight, lovasz_weight = 0.4, 0.3, 0.3
        else:
            focal_weight, dice_weight, lovasz_weight = 0.3, 0.3, 0.4

        alpha = torch.log(1.0 + self.class_weight)
        alpha = alpha / alpha.max()

        # Calculate semantic loss
        # semantic_ce_loss = self.semantic_ce_loss(semantic_pred, semantic_gt)
        # semantic_ce_loss = F.cross_entropy(semantic_pred, semantic_gt, weight=self.class_weight, ignore_index=self.ignore_label, reduction='mean')
        semantic_focal_loss = self.semantic_focal_loss(semantic_pred, semantic_gt, alpha)
        semantic_lovasz_loss = self.semantic_lovasz_loss(F.softmax(semantic_pred, dim=1), semantic_gt)
        semantic_dice_loss = self.semantic_dice_loss(semantic_pred, semantic_gt)

        semantic_loss = dice_weight * semantic_dice_loss + focal_weight * semantic_focal_loss + lovasz_weight * semantic_lovasz_loss

        # Calculate other loss
        dynamic_ce_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
        # dynamic_dice_loss = self.dynamic_dice_loss(dynamic_pred, dynamic_gt)
        # dynamic_loss = self.coe[0] * dynamic_dice_loss + self.coe[1] * dynamic_ce_loss

        lane_ce_loss = self.loss_func_static(lane_pred, lane_gt)
        # lane_dice_loss = self.static_dice_loss(lane_pred, lane_gt)
        # lane_loss = self.coe[0] * lane_dice_loss + self.coe[1] * lane_ce_loss

        total_loss = self.weights[0] * semantic_loss + self.weights[1] * dynamic_ce_loss + self.weights[2] * lane_ce_loss

        # loss_weights = torch.softmax(self.weights, dim=0)
        # total_loss = loss_weights[0] * semantic_loss + loss_weights[1] * dynamic_ce_loss + loss_weights[2] * lane_ce_loss

        # semantic_weight, dynamic_weight, lane_weight = 0.2, 0.21*(semantic_loss/(dynamic_ce_loss+1e-10)), 0.21*(semantic_loss/(lane_ce_loss+1e-10))
        # total_loss = semantic_weight * semantic_loss + dynamic_weight * dynamic_ce_loss + lane_weight * lane_ce_loss

        self.loss_dict.update({'Dynamic Loss': dynamic_ce_loss,
                               'Lane Loss': lane_ce_loss,
                               'Semantic Loss': semantic_loss,
                               'Total_loss': total_loss,
                               'semantic_lovasz_loss': semantic_lovasz_loss,
                               'semantic_dice_loss': semantic_dice_loss,
                               # 'semantic_ce_loss': semantic_ce_loss,
                               'semantic_focal_loss': semantic_focal_loss,
                               })

        return self.loss_dict['Total_loss']


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['Total_loss']
        semantic_loss = self.loss_dict['Semantic Loss']
        lane_loss = self.loss_dict['Lane Loss']
        dynamic_loss = self.loss_dict['Dynamic Loss']
        lovasz_loss = self.loss_dict['semantic_lovasz_loss']
        dice_loss = self.loss_dict['semantic_dice_loss']
        # ce_loss = self.loss_dict['semantic_ce_loss']
        focal_loss = self.loss_dict['semantic_focal_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Total Loss: %.4f || Semantic Loss: %.4f || Dynamic Loss: %.4f || Lane Loss: %.4f" %
                  (epoch, batch_id + 1, batch_len, total_loss.item(), semantic_loss.item(), dynamic_loss.item(), lane_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Total Loss: %.4f || Semantic Loss: %.4f || Dynamic Loss: %.4f || Lane Loss: %.4f" %
                                 (epoch, batch_id + 1, batch_len, total_loss.item(), semantic_loss.item(), dynamic_loss.item(), lane_loss.item()))

        # if pbar is None:
        #     print("[epoch %d][%d/%d], || Total Loss: %.4f || Lovasz Loss: %.4f || CE Loss: %.4f || Focal Loss: %.4f" %
        #           (epoch, batch_id + 1, batch_len, total_loss.item(), lovasz_loss.item(), ce_loss.item(), focal_loss.item()))
        # else:
        #     pbar.set_description("[epoch %d][%d/%d], || Total Loss: %.4f || Lovasz Loss: %.4f || CE Loss: %.4f || Focal Loss: %.4f" %
        #                          (epoch, batch_id + 1, batch_len, total_loss.item(), lovasz_loss.item(), ce_loss.item(), focal_loss.item()))

        # writer.add_scalar('Total Seg Loss (Training Sample)', total_loss.item(), epoch*batch_len + batch_id)
        writer.add_scalars('sample_loss',
                           {'Total Loss': total_loss.item(), 'Semantic Loss': semantic_loss.item(),
                            'Dynamic Loss': dynamic_loss.item(), 'Lane Loss': lane_loss.item()}, epoch * batch_len + batch_id)
        # writer.add_scalar('Semantic Loss (Training Sample)', semantic_loss.item(), epoch * batch_len + batch_id)
        # writer.add_scalar('Dynamic Loss (Training Sample)', dynamic_loss.item(), epoch * batch_len + batch_id)
        # writer.add_scalar('Lane Loss (Training Sample)', lane_loss.item(), epoch * batch_len + batch_id)

        # writer.add_scalar('Samples', 'Semantic Lovasz Loss', lovasz_loss.item(), epoch * batch_len + batch_id)
        # writer.add_scalar('Samples', 'Semantic CE Loss', ce_loss.item(), epoch * batch_len + batch_id)
        # writer.add_scalar('Samples', 'Semantic Focal Loss', focal_loss.item(), epoch * batch_len + batch_id)
        writer.add_scalars('semantic_loss', {'Semantic Dice Loss': dice_loss.item(), 'Semantic Focal Loss': focal_loss.item(), 'Semantic lovasz Loss': lovasz_loss.item()}, epoch * batch_len + batch_id)
        # writer.add_scalars('Head_Weights',
        #                    {'Semantic weight': self.loss_dict['head_weights'][0].item(), 'Dynamic weight': self.loss_dict['head_weights'][1].item(),
        #                     'Lane weight': self.loss_dict['head_weights'][2].item()}, epoch * batch_len + batch_id)
        # writer.add_scalars('Class Weights', {f'Class_{i}': class_weights[i] for i in range(len(class_weights))}, epoch * batch_len + batch_id)
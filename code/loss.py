import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from skimage.measure import label, regionprops


def sim(x, y):
    '''
    compute dot product similarity
    :param x: (1,  hidden)
    :param y: (batch,  hidden)
    '''
    normx = torch.linalg.norm(x, dim=-1)
    normy = torch.linalg.norm(y, dim=-1)
    x_norm = x / (normx.unsqueeze(-1) + 1e-8)
    y_norm = y / (normy.unsqueeze(-1) + 1e-8)
    return torch.matmul(x_norm, y_norm.T)


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


def dice_coieffience_loss(mask_logit, mask_gt, eps = 1e-12):
    '''
    :param mask_logit: before softmax, B, H, W
    :param mask_gt: B, H, W
    :return:
    '''
    mask_pred = mask_logit.sigmoid()
    mask_pred = mask_pred - mask_pred.amin(dim=[1, 2], keepdim=True)  # H, W min
    mask_pred = mask_pred / mask_pred.amax(dim=[1, 2], keepdim=True)
    intersection = (mask_pred * mask_gt).sum(dim=[1, 2])
    union = eps + mask_pred.sum(dim=[1, 2]) + mask_gt.sum(dim=[1, 2])
    loss = 1. -2 * intersection / union
    loss = loss.mean()
    return loss


def cross_entropy_prob_loss(mask_logit, mask_gt, wmask=None):
    mask_logit = mask_logit.view(-1)
    mask_gt = mask_gt.view(-1)
    mask_gt_weight = torch.where(mask_gt > 0., 1, 0)
    neg_weight = mask_gt_weight.sum() / mask_gt_weight.size(0)
    pos_weight = 1 - mask_gt_weight.sum() / mask_gt_weight.size(0)
    weights = torch.where(mask_gt_weight > 0., pos_weight, neg_weight)
    loss_bce = F.binary_cross_entropy_with_logits(
        mask_logit.squeeze(),
        mask_gt.squeeze(),
        reduction='none'
    )
    loss_bce = loss_bce * weights
    loss_bce = loss_bce.mean()
    return loss_bce


class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'

    def __init__(self, alpha=0.25, gamma=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred_logits = pred_logits.view(-1)
        target = target.view(-1)
        target_weight = torch.where(target > 0., 1, 0)
        neg_weight = target_weight.sum() / target_weight.size(0)
        pos_weight = 1 - target_weight.sum() / target_weight.size(0)
        alpha = torch.where(target_weight > 0., pos_weight, neg_weight)

        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        losses = alpha * ((1. - pt) ** self.gamma) * ce
#         losses = alpha * ce
        return losses.mean()


# class main_loss(nn.Module):
#     def __init__(self, temperature=0.5):
#         super().__init__()
#         self.t = temperature

#     def forward(self, mask_logit, mask_gt, wmask=None):
#         ce_loss = cross_entropy_prob_loss(mask_logit, mask_gt.float().squeeze(-1), wmask)
#         dc_loss = dice_coieffience_loss(mask_logit, mask_gt.float().squeeze(-1))
#         loss = {'ce_loss': ce_loss,
#                 'dc_loss': dc_loss}
#         return loss
    
    
class main_loss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.focal = FocalLoss()

    def forward(self, mask_logit, mask_gt, wmask=None):
        focal_loss = self.focal(mask_logit, mask_gt.float().squeeze(-1))
        dc_loss = dice_coieffience_loss(mask_logit, mask_gt.float().squeeze(-1))
        loss = {'ce_loss': focal_loss,
                'dc_loss': dc_loss}
        return loss


if __name__ == '__main__':
    pred = torch.randn(size=(8, 32, 32, 1))
    target = torch.randn(size=(8, 32, 32, 1))
    mask_gt = torch.where(target > 0, 1., 0.)

    loss_fn = main_loss()
    res = loss_fn(pred, mask_gt)
    print(res.shape)
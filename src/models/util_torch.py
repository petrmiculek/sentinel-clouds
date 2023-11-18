import numpy as np
import torch
from torch import Tensor
import segmentation_models_pytorch as smp

class EarlyStopping:
    """
        Early Stopping adapted from: vvv

        Early stopping is used to avoid overfitting of the model.
        As the PyTorch library does not contain built-in early stopping, this class is from following repository:
        https://github.com/Bjarten/early-stopping-pytorch
        Original author: Bjarte Mehus Sunde, 2018,  BjarteSunde@outlook.com
        Licence: MIT License, Copyright (c) 2018 Bjarte Mehus Sunde
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """
    def __init__(self, patience=10, verbose=False, delta=1e-4, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).')
        if self.path is not None:
            torch.save(model.state_dict(), self.path)
            self.trace_func("Checkpoint saved")
        self.val_loss_min = val_loss

class DiceLoss(torch.nn.Module):
    def __init__(self, one_hot=False) -> None:
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.one_hot = one_hot
        self.dims = (1, 2, 3)  # TODO parameterize for batch x sample version

    def forward(self, pred, target):
        if self.one_hot:
            # not needed for binary labels 
            # ... but my impl will behave differently than the multiclass original
            target = torch.stack([1 - target, target], dim=1)
            pred = torch.stack([1 - pred, pred], dim=1)

        intersection = torch.sum(pred * target, self.dims)
        union = torch.sum(pred + target, self.dims)
        dice_score = (2 * intersection) / (union + self.eps)
        return torch.mean(1 - dice_score)  # score to loss

class JaccardLoss(torch.nn.Module):
    """ Inspired by: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py """
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.eps = 1e-6
        self.dims = (1, 2, 3)
    
    def forward(self, pred, target):
        intersection = torch.sum(pred * target, self.dims)
        union = torch.sum(pred + target, self.dims) - intersection
        jaccard_score = (intersection + self.eps) / (union + self.eps)
        return torch.mean(1 - jaccard_score)

class MCCLoss(torch.nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    Source: https://github.com/kakumarabhishek/MCC-Loss
    """
    def __init__(self):
        super(MCCLoss, self).__init__()
        self.smooth = 1

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        TODO this is per-batch, not per-sample
        TODO TP/FP/.. feel wrong for a one-hot encoding - which I don't give it anyway
        """
        tp = torch.sum(inputs * targets)
        tn = torch.sum((1 - inputs) * (1 - targets))
        fp = torch.sum(inputs * (1 - targets))
        fn = torch.sum((1 - inputs) * targets)
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt(
            (tp + fp + self.smooth) * 
            (tp + fn + self.smooth) * 
            (tn + fp + self.smooth) * 
            (tn + fn + self.smooth))
        mcc = numerator.sum() / (denominator.sum() + 1.0)
        return 1 - mcc

class DiceAndBCELogitLoss(torch.nn.Module):
    def __init__(self, bce_factor=1, dice_factor=1, choice='dice', pos_weight=None, one_hot=False) -> None:
        super(DiceAndBCELogitLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if choice == 'dice':
            self.dice = DiceLoss(one_hot=one_hot)
        elif choice == 'jaccard':
            self.dice = JaccardLoss()
        elif choice == 'mcc':
            self.dice = MCCLoss()
        elif choice == 'focal':
            self.dice = smp.losses.FocalLoss(mode='binary', alpha=0.5, gamma=2, reduction='mean')
        
        self.dice = MCCLoss()
        self.bce_losses = []
        self.dice_losses = []
        self.bce_factor = bce_factor
        self.dice_factor = dice_factor
        self.pos_weight = pos_weight
    
    def forward(self, logits, target):
        bce_loss = self.bce(logits, target) * self.bce_factor
        pred_sig = torch.sigmoid(logits)
        dice_loss = self.dice(pred_sig, target) * self.dice_factor
        self.bce_losses.append(bce_loss.item())
        self.dice_losses.append(dice_loss.item())
        return bce_loss + dice_loss

# Versions below unused:
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # input and target shape is BxCxHxW
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def dl(pred, target):
        dims = (1, 2, 3)
        intersection = torch.sum(pred * target, dims)
        union = torch.sum(pred + target, dims)
        dice_score = 2. * intersection / (union + 1e-6)
        return torch.mean(1. - dice_score)

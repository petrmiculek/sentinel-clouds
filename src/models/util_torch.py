import numpy as np
import torch
from torch import Tensor

class EarlyStopping:
    """
        Early Stopping adapted from: vvvvvvv

        Early stopping is used to avoid overfitting of the model.
        As the PyTorch library does not contain built-in early stopping, this class is from following repository:
        https://github.com/Bjarten/early-stopping-pytorch

        Original author:
        Bjarte Mehus Sunde, 2018

        Original author's mail:
        BjarteSunde@outlook.com

        Licence:
        MIT License

        Copyright (c) 2018 Bjarte Mehus Sunde

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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# import one_hot
class DiceLoss(torch.nn.Module):
    def __init__(self, eps1=1e-6, eps2=1e-6, one_hot=False) -> None:
        super(DiceLoss, self).__init__()
        self.eps1: float = eps1
        self.eps2: float = eps2
        self.one_hot = one_hot

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not len(pred.shape) == 4:
            raise ValueError("Invalid pred shape, we expect BxNxHxW. Got: {}"
                             .format(pred.shape))
        if not pred.shape[-2:] == target.shape[-2:]:
            raise ValueError("pred and target shapes must be the same. Got: {}"
                             .format(pred.shape, pred.shape))
        if not pred.device == target.device:
            raise ValueError(
                "pred and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))

        # create the labels one hot tensor
        if self.one_hot:
            # not needed for binary labels 
            # ... but my impl will behave differently than the multiclass original
            target = torch.stack([1 - target, target], dim=1)
            pred = torch.stack([1 - pred, pred], dim=1)

        # compute the dice score
        dims = (1, 2, 3)
        intersection = torch.sum(pred * target, dims)
        union = torch.sum(pred + target, dims)
        dice_score = (2 * intersection + self.eps1) / (union + self.eps2)
        return torch.mean(1 - dice_score)  # score to loss

class DiceAndBCELogitLoss(torch.nn.Module):
    def __init__(self, bce_factor=1, dice_factor=1, pos_weight=None, eps1=1e-6, eps2=1e-6, one_hot=False) -> None:
        super(DiceAndBCELogitLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(eps1=eps1, eps2=eps2, one_hot=one_hot)
        self.bce_losses = []
        self.dice_losses = []
        self.bce_factor = bce_factor
        self.dice_factor = dice_factor
        self.pos_weight = pos_weight
    
    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred)
        bce_loss = self.bce(pred, target) * self.bce_factor
        dice_loss = self.dice(pred_sig, target) * self.dice_factor
        self.bce_losses.append(bce_loss.item())
        self.dice_losses.append(dice_loss.item())
        return bce_loss + dice_loss

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


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def dl(pred, target):
        dims = (1, 2, 3)
        intersection = torch.sum(pred * target, dims)
        union = torch.sum(pred + target, dims)

        dice_score = 2. * intersection / (union + 1e-6)
        return torch.mean(1. - dice_score)

import torch
import segmentation_models_pytorch as smp

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
        else:
            raise ValueError(f"Invalid loss choice {choice}")
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

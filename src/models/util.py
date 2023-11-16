import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score

def lod_mean(list_of_dicts):
    """ Per-key mean of a list of dictionaries. """
    return {k: np.mean([i[k] for i in list_of_dicts]) for k in list_of_dicts[0].keys()}

def compute_metrics(labels, preds, use_th=True, suffix=None):
    labels, preds = labels.flatten(), preds.flatten()
    if use_th:
        preds = preds > 0.5
    res = {'Accuracy': accuracy_score(labels, preds),
            'Precision': precision_score(labels, preds),
            'Recall': recall_score(labels, preds),
            'F1': f1_score(labels, preds),
            'IoU': jaccard_score(labels, preds)}
    if suffix is not None:
        res = keys_append(res, suffix)
    return res

def compute_metrics_own(labels, preds, use_th=True, suffix=None):
    eps = 1e-7
    if use_th:
        preds = preds > 0.5
    tp = torch.sum((preds == 1) & (labels == 1), dim=(1, 2, 3))
    tn = torch.sum((preds == 0) & (labels == 0), dim=(1, 2, 3))
    fp = torch.sum((preds == 1) & (labels == 0), dim=(1, 2, 3))
    fn = torch.sum((preds == 0) & (labels == 1), dim=(1, 2, 3))
    # dim-s account for samples in batch

    dice = (2 * tp) / (2 * tp + fp + fn + eps)  # intersection twice in the denominator
    iou = tp / (tp + fp + fn + eps)  # == Jaccard score; intersection only once in the denominator
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    res = {'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Dice': dice,
            'F1': f1,
            'IoU': iou}
    
    # average over batch
    res = {k: torch.mean(v).item() for k, v in res.items()}
    
    if suffix is not None:
        res = keys_append(res, suffix)
    return res


def evaluate_metrics(model, loader, criterion=None, suffix=None, device='cuda'):
    model.eval()
    loss = 0
    # labels, preds = [], []
    metrics = []
    with torch.no_grad():
        for s in tqdm(loader):
            x, y = s['image'].to(device), s['label'].to(device)
            logits = model(x)
            if criterion is not None:
                loss += criterion(logits, y).item()
            pred = torch.sigmoid(logits)
            # labels.append(y.cpu().numpy())
            # preds.append(pred.cpu().numpy())
            metrics.append(compute_metrics_own(y, pred))
    
    metrics = {**lod_mean(metrics), "Loss": loss / len(loader)}
    if suffix is not None:
        metrics = keys_append(metrics, suffix)
    return metrics

def evaluate_get_preds(model, loader, criterion=None, device='cuda'):
    model.eval()
    loss = 0
    labels, preds = [], []
    with torch.no_grad():
        for s in tqdm(loader):
            x, y = s['image'].to(device), s['label'].to(device)
            logits = model(x)
            if criterion is not None:
                loss += criterion(logits, y).item()
            pred = torch.sigmoid(logits)
            labels.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    return {"Loss": loss / len(loader),
            "Labels": np.concatenate(labels),
            "Preds": np.concatenate(preds)}

def keys_append(dictionary, suffix):
    """Appends suffix to all keys in dictionary."""
    return {k + suffix: v for k, v in dictionary.items()}

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

class DiceLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

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
        # compute softmax over the classes axis
        # - model outputs are already softmax-ed

        # create the labels one hot tensor
        # target_one_hot = one_hot(target, num_classes=pred.shape[1],
                                #  device=pred.device, dtype=pred.dtype)
        # - Binary labels => not needed. ... but my impl will behave differently

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(pred * target, dims)
        union = torch.sum(pred + target, dims)

        dice_score = 2. * intersection / (union + self.eps)
        return torch.mean(1. - dice_score)
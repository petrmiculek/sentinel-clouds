import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score

def evaluate(model, loader, criterion=None, device='cuda'):
    model.eval()
    loss = 0
    labels, preds = [], []
    with torch.no_grad():
        for s in tqdm(loader):
            x, y = s['image'].to(device), s['label'].to(device)
            pred = model.predict(x)
            if criterion is not None:
                loss += criterion(pred, y).item()
            labels.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    return {"loss": loss / len(loader),
            "labels": np.concatenate(labels),
            "preds": np.concatenate(preds)}

def compute_metrics(labels, preds, use_th=True):
    labels, preds = labels.flatten(), preds.flatten()
    if use_th:
        preds = preds > 0.5
    return {'Accuracy': accuracy_score(labels, preds),
            'Precision': precision_score(labels, preds),
            'Recall': recall_score(labels, preds),
            'F1': f1_score(labels, preds),
            'IoU': jaccard_score(labels, preds)}

"""
    :filename util.py (originally EarlyStopping.py)

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
class EarlyStopping:
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

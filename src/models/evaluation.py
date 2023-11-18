from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score
import numpy as np
import torch
from tqdm.auto import tqdm
from src.models.util import keys_append, lod_mean

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
    labels = labels > 0.5
    dims = (1, 2, 3)
    tp = torch.sum((preds == 1) & (labels == 1), dim=dims)
    tn = torch.sum((preds == 0) & (labels == 0), dim=dims)
    fp = torch.sum((preds == 1) & (labels == 0), dim=dims)
    fn = torch.sum((preds == 0) & (labels == 1), dim=dims)
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
        progress_bar = tqdm(loader)
        for i, s in enumerate(progress_bar, start=1):
            x, y = s['image'].to(device), s['label'].to(device)
            logits = model(x)
            if criterion is not None:
                loss += criterion(logits, y).item()
                progress_bar.set_postfix(loss=f'{loss / i:.4f}', refresh=False)
            pred = torch.sigmoid(logits)
            # labels.append(y.cpu().numpy())
            # preds.append(pred.cpu().numpy())
            metrics.append(compute_metrics_own(y, pred))
        loss = loss / len(loader)
        progress_bar.set_postfix(loss=f'{loss:.4f}', refresh=True)

    
    metrics = {**lod_mean(metrics), "Loss": loss}
    if suffix is not None:
        metrics = keys_append(metrics, suffix)
    return metrics

def evaluate_get_preds(model, loader, criterion=None, device='cuda'):
    model.eval()
    loss = 0
    labels, preds = [], []
    with torch.no_grad():
        progress_bar = tqdm(loader)
        for s in progress_bar:
            x, y = s['image'].to(device), s['label'].to(device)
            logits = model(x)
            if criterion is not None:
                loss += criterion(logits, y).item()
                progress_bar.set_postfix(loss=f'{loss:.4f}', refresh=False)
            pred = torch.sigmoid(logits)
            labels.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())
    return {"Loss": loss / len(loader),
            "Labels": np.concatenate(labels),
            "Preds": np.concatenate(preds)}
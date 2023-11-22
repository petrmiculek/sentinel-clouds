""" Train a model. """
# %% Imports
import sys
# remove last element from path
import os
from os.path import abspath, join, exists
sys.path.pop()  # preexisting imports path messing up imports
sys.path.append(abspath(join('..')))  # ,'src'
sys.path.append(abspath(join('../..')))  # ,'src'
# print("Python imports path:", "\n".join(sys.path))
# %%
# standard library
import argparse
from copy import deepcopy
from types import SimpleNamespace
# external
import numpy as np
import torch
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import sigmoid
from torch.cuda.amp import GradScaler
from torch import autocast, sigmoid
from tqdm.auto import tqdm
from torchinfo import summary
from timm.scheduler import CosineLRScheduler  # TODO unused
import wandb as wb
import segmentation_models_pytorch as smp
# local
from src.data.dataset import get_loaders, get_loaders_processed
from src.models.unet import UNet
from src.models.util_torch import EarlyStopping, DiceLoss, dice_loss, DiceAndBCELogitLoss
from src.models.evaluation import evaluate_metrics, compute_metrics_own
from src.models.util import keys_append, print_dict
from src.visualization.visualize import plot_many

# set numpy/torch print precision .4f
np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

# args
use_pos_weight = False
path_data = '/mnt/sdb1/code/sentinel2/processed'
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=path_data, help='path to dataset')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
parser.add_argument('--bce', type=float, default=1, help='bce loss factor')
parser.add_argument('--dice', type=float, default=0, help='dice loss factor')
parser.add_argument('--backbone', type=str, default='resnet18', help="backbone architecture, e.g. resnet18, timm-res2net50_26w_4s, timm-regnetx_002")
parser.add_argument('--arch', type=str, default='Unet', choices=['Unet', 'UnetPlusPlus', 'DeepLabV3Plus'], help='architecture')
parser.add_argument('--no_log', '-n', action='store_true', help='disable logging to wandb')
parser.add_argument('--weigh_bce', '-w', action='store_true', help='use bce loss weighting')
parser.add_argument('--loss', '-l', type=str, default='dice', choices=['dice', 'jaccard', 'mcc', 'focal'], help='2nd loss used (except bce)')
parser.add_argument('--workers', type=int, default=1, help='number of dataloader workers')
parser.add_argument('--optim', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'], help='optimizer')
args = parser.parse_args()
if args.data == 'meta':
    args.data = '/storage/brno2/home/petrmiculek/sentinel2/processed'
# %% Hyperparameters
wb.init(project="clouds", mode='disabled' if args.no_log else None)
cfg = wb.config
''' Preprocessing '''
cfg.tile_size = 224
cfg.crop_pad_mask = 'crop'
''' Data '''
cfg.workers = args.workers
cfg.batch_size = args.batch_size
''' Model '''
cfg.backbone = args.backbone
cfg.architecture = args.arch
''' Training '''
cfg.epochs = args.epochs
cfg.lr = args.lr
cfg.loss = args.loss
cfg.bce_factor = args.bce
cfg.dice_factor = args.dice
cfg.optim = args.optim
run_name = wb.run.name
outputs_dir = join('runs', run_name)
os.makedirs(outputs_dir, exist_ok=True)
checkpoint_path = join(outputs_dir, 'model_checkpoint.pt') if not args.no_log else None
# %% Dataset
dataset_kwargs = {'tile_size': cfg.tile_size, 'crop_pad_mask': cfg.crop_pad_mask}  # doesn't matter when using processed data
loader_kwargs = {'batch_size': cfg.batch_size, 'num_workers': cfg.workers, 'pin_memory': True, 'shuffle': True}
loader = get_loaders_processed(args.data, splits=['train', 'val'], **dataset_kwargs, **loader_kwargs)
# loader['train'] = loader['test']  # TODO: debug, remove
# %% Model + training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet(in_channels=4).to(device)  # unused, own implementation did not train well
model = getattr(smp, cfg.architecture)(encoder_name=cfg.backbone, encoder_weights=None, in_channels=4, classes=1).to(device)
model_summary = summary(model, input_size=(1, 4, 224, 224))
# test model forward pass
assert model(torch.randn(1, 4, 224, 224, device=device)).shape == (1, 1, 224, 224)
train_mean = loader['train'].dataset.labels.mean()
cfg.pos_weight = (3 - 2*train_mean) / (4*train_mean + 1e-1) if use_pos_weight else None
criterion = DiceAndBCELogitLoss(cfg.bce_factor, cfg.dice_factor, choice=cfg.loss, pos_weight=cfg.pos_weight)
optimizer = getattr(torch.optim, cfg.optim)(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
scaler = GradScaler()
early_stopping = EarlyStopping(patience=40, path=checkpoint_path)  # don't stop training, save best model
# scheduler = ReduceLROnPlateau(optimizer, patience=20)
epoch_steps = len(loader['train'])
scheduler = CosineLRScheduler(optimizer, t_initial=cfg.epochs * epoch_steps, warmup_t=epoch_steps * 5, warmup_lr_init=1e-5, lr_min=2e-8, cycle_decay=1e-3)  # cycle_mul=3, cycle_limit=3
cfg.update({'scheduler': scheduler.__class__.__name__, 'train_size': len(loader['train'].dataset),
    'model_bytes': model_summary.total_param_bytes, 'model_params': model_summary.total_params})
# %% Training
best_accu_val = 0
best_res = None
epochs_trained = 0
stop_training = False
grad_acc_steps = 1
for epoch in range(epochs_trained, epochs_trained + cfg.epochs):
    model = model.train()
    ep_train_loss = 0
    metrics_train = []
    criterion.bce_losses, criterion.dice_losses = [], []
    ep_grad = 0
    print(f"lr: {optimizer.param_groups[0]['lr']:.4e}")
    try:
        progress_bar = tqdm(loader['train'], mininterval=1., desc=f'ep{epoch} train')
        for i, sample in enumerate(progress_bar, start=1):
            img, label = sample['image'].to(device, non_blocking=True), sample['label'].to(device, non_blocking=True)
            # forward pass
            logits = model(img)
            loss = criterion(logits, label)
            # backward pass
            loss.backward()
            ep_grad += torch.nn.utils.clip_grad_norm_(model.parameters(), 20, error_if_nonfinite=True).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():  # save predictions
                ep_train_loss += loss.cpu().numpy()
                pred = sigmoid(logits)
                metrics_train.append(compute_metrics_own(label, pred))
                progress_bar.set_postfix(loss=f'{loss:.4f}', refresh=False)
            scheduler.step(i + epoch * epoch_steps)
        # end of training epoch loop
    except KeyboardInterrupt:
        print(f'Ctrl+C stopped training')
        stop_training = True
    # compute training metrics
    ep_train_loss /= len(loader['train'])
    ep_grad /= len(loader['train'])
    ep_bce = np.mean(criterion.bce_losses)
    ep_dice = np.mean(criterion.dice_losses)
    metrics_train = {k: np.mean([m[k] for m in metrics_train]) for k in metrics_train[0].keys()}
    metrics_train = keys_append(metrics_train, ' Training')
    ''' Validation loop '''
    model = model.eval()
    metrics_val = evaluate_metrics(model, loader['val'], criterion, suffix=' Validation')
    # log results
    res_epoch = {'Loss Training': ep_train_loss, 'Grad Training': ep_grad,
                 'BCELoss Training': ep_bce, 'DiceLoss Training': ep_dice,
                   **metrics_train, **metrics_val, 'Learning Rate': optimizer.param_groups[0]['lr']
                   }
    print_dict(res_epoch, title=f'Epoch {epoch}')
    wb.log(res_epoch, step=epoch)
    if metrics_val['Accuracy Validation'] >= best_accu_val:  # save best results
        best_accu_val = metrics_val['Accuracy Validation']
        best_res = deepcopy(res_epoch)
    epochs_trained += 1
    # scheduler.step(metrics_val['Loss Validation'])  # used for ReduceLROnPlateau
    early_stopping(metrics_val['Loss Validation'], model)  # model checkpointing
    if early_stopping.early_stop or stop_training:
        print('Early stopping')
        break
cfg.update({'epochs': epochs_trained}, allow_val_change=True)
# %% Export model
print_dict(best_res, title='Best epoch:')
torch.save(model.state_dict(), join(outputs_dir, 'model.pt'))  # also save last epoch
print(f'Saved model to {outputs_dir}')
if False:
    # save model as onnx
    dummy_input = torch.randn(1, 4, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, opset_version=11, input_names=['input'], output_names=['output'])

# %% Save a batch of predictions to W&B tables
table = wb.Table(columns=['Image', 'Label', 'Prediction'])
class_labels = {1: 'Clouds'}
for i, _ in enumerate(img):
    # convert image to numpy hwc rgb, label kept single-channel
    img_i = img[i].detach().cpu().numpy().transpose((1, 2, 0))[:3]
    label_i = label[i, 0].detach().cpu().numpy() > 0.5
    pred_i = pred[i, 0].detach().cpu().numpy() > 0.5
    table.add_data(wb.Image(img_i, masks={"prediction": {"mask_data": pred_i, "class_labels": class_labels}, 
                                          "ground_truth": {"mask_data": label_i, "class_labels": class_labels}}), 
                                          wb.Image(label_i), wb.Image(pred_i))
wb.log({'Training Batch': table})

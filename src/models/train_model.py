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
from copy import deepcopy
from types import SimpleNamespace
# external
import numpy as np
import torch
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import sigmoid
from torch.cuda.amp import GradScaler
from torch import autocast, sigmoid
from tqdm.auto import tqdm
from torchinfo import summary
# from timm.scheduler import CosineLRScheduler  # TODO unused
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
# path_data = '/storage/brno2/home/petrmiculek/sentinel2/processed'

# %%
# Hyperparameters
wb.init(project="clouds")  # mode='disabled'
cfg = wb.config
''' Preprocessing '''
cfg.tile_size = 224
cfg.crop_pad_mask = 'crop'
# -
''' Data '''
cfg.workers = 0
cfg.batch_size = 1
''' Model '''
# -
''' Training '''
cfg.epochs = 5
cfg.lr = 5e-3
cfg.bce_factor=1
cfg.dice_factor=1
# cfg.warmup_prop = 0.1  # TODO lr warmup
# wb.define_metric("batches")
# wb.define_metric("Training Loss", step_metric='batches')
run_name = wb.run.name
outputs_dir = join('runs', run_name)
os.makedirs(outputs_dir, exist_ok=True)
checkpoint_path = join(outputs_dir, 'model_checkpoint.pt')
# %%
dataset_kwargs = {'tile_size': cfg.tile_size, 'crop_pad_mask': cfg.crop_pad_mask}  # doesn't matter when using processed data
loader_kwargs = {'batch_size': cfg.batch_size, 'num_workers': cfg.workers, 'pin_memory': True}
loader = get_loaders_processed(path_data, splits=['test', 'val'], **dataset_kwargs, **loader_kwargs)
loader['train'] = loader['test']  # TODO: debug, remove
# %% Model + training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet(in_channels=4).to(device)
model = smp.Unet(encoder_name="resnet18",  # encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,in_channels=4,classes=1).to(device)
model_summary = summary(model, input_size=(1, 4, 224, 224))
# %%
# print(model_summary)  # already printed in the summary call
train_mean = loader['train'].dataset.labels.mean()
cfg.pos_weight = (3 - 2*train_mean) / (4*train_mean + 1e-1) if use_pos_weight else 1

criterion = DiceAndBCELogitLoss(cfg.bce_factor, cfg.dice_factor)  # , pos_weight=pos_weight)
optimizer = Adam(model.parameters(), lr=cfg.lr)  # , weight_decay=1e-2
scaler = GradScaler()  # mixed precision training (16-bit)
early_stopping = EarlyStopping(patience=50, path=checkpoint_path)  # TODO 50 debug
scheduler = ReduceLROnPlateau(optimizer, patience=20)
# scheduler = CosineLRScheduler(optimizer, t_initial=epoch_steps // 3, warmup_t=warmup_steps, warmup_lr_init=1e-6, lr_min=2e-8, cycle_decay=0.7, cycle_mul=3, cycle_limit=3)
# wb_watch_freq = 100
# wb.watch(model, criterion, log='gradients', log_freq=wb_watch_freq)
cfg.update({'criterion': criterion.__class__.__name__,'optimizer': optimizer.__class__.__name__,
                   'scheduler': scheduler.__class__.__name__,'architecture': model.__class__.__name__,
                    'train_size': len(loader['train'].dataset),'model_bytes': model_summary.total_param_bytes,
                   'model_params': model_summary.total_params})
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
    try:
        progress_bar = tqdm(loader['train'], mininterval=1., desc=f'ep{epoch} train')
        for i, sample in enumerate(progress_bar, start=1):
            img, label = sample['image'].to(device, non_blocking=True), sample['label'].to(device, non_blocking=True)
            # forward pass
            # with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(img)  # prediction
            loss = criterion(logits, label)
            # backward pass
            loss.backward()
            # scaler.scale(loss).backward()
            # clip gradients
            ep_grad += torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True).item()
            # if i % grad_acc_steps == 0:  # gradient step with accumulated gradients
                # scaler.step(optimizer)
                # scaler.update()
                # optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():  # save predictions
                ep_train_loss += loss.cpu().numpy()
                pred = sigmoid(logits)  # is torch.float16
                metrics_train.append(compute_metrics_own(label, pred))
            progress_bar.set_postfix(loss=f'{loss:.4f}', refresh=False)
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
                   **metrics_train, **metrics_val
                   }
    print_dict(res_epoch, title=f'Epoch {epoch}')
    wb.log(res_epoch, step=epoch)
    if metrics_val['Accuracy Validation'] >= best_accu_val:  # save best results
        best_accu_val = metrics_val['Accuracy Validation']
        best_res = deepcopy(res_epoch)
    epochs_trained += 1
    scheduler.step(metrics_val['Loss Validation'])  # LR scheduler
    early_stopping(metrics_val['Loss Validation'], model)  # model checkpointing
    if early_stopping.early_stop or stop_training:
        print('Early stopping')
        break
# %% Eval
# metrics_val = evaluate_metrics(model, loader['val'], criterion, device)
# metrics_val
# %% Export model
if False:
    # save model as .pt
    torch.save(model.state_dict(), 'model.pt')
    # save model as onnx
    dummy_input = torch.randn(1, 4, 224, 224, device='cuda')
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, opset_version=11, input_names=['input'], output_names=['output'])

# %% Save a batch of predictions to W&B tables
table = wb.Table(columns=['Image', 'Label', 'Prediction'])
for i, _ in enumerate(img):
    # convert image to numpy hwc, label to rgb
    img_i = img[i].detach().cpu().numpy().transpose((1, 2, 0))
    label_i = label[i].detach().cpu().numpy() > 0.5
    pred_i = pred[i].detach().cpu().numpy() > 0.5
    table.add_data(wb.Image(img_i), wb.Image(label_i), wb.Image(pred_i))
wb.log({'Training Batch': table})

# %% Loss experimenting
if False:
    lb = BCEWithLogitsLoss()
    lbb = BCELoss()
    ld = DiceLoss()
    lbd = DiceAndBCELogitLoss()

    t = sample['label'].to(device)

    log0 = torch.zeros_like(t).to(device) - 2.9444
    log1 = torch.ones_like(t).to(device) + 2.9444
    inverse_sigmoid = lambda x: -torch.log(1 / x - 1)
    logt = inverse_sigmoid(t)
    assert torch.allclose(t, sigmoid(logt))

    p0 = torch.zeros_like(t).to(device)
    p1 = torch.ones_like(t).to(device)

    arr = np.array([[lb(log0, t).item(), lb(log1, t).item(), lb(logt, t).item()],
                    [ld(p0, t).item(), ld(p1, t).item(), ld(t, t).item()],
                    [lbd(log0, t).item(), lbd(log1, t).item(), lbd(logt, t).item()]])
    print(arr)

    t0 = torch.zeros_like(t).to(device) + 0.05
    t1 = torch.ones_like(t).to(device) - 0.05

    at0 = np.array([[lb(log0, t0).item(), lb(log1, t0).item(), lb(logt, t0).item()],
                    [ld(p0, t0).item(), ld(p1, t0).item(), ld(t, t0).item()],
                    [lbd(log0, t0).item(), lbd(log1, t0).item(), lbd(logt, t0).item()]])
    print(at0)

    at1 = np.array([[lb(log0, t1).item(), lb(log1, t1).item(), lb(logt, t1).item()],
                    [ld(p0, t1).item(), ld(p1, t1).item(), ld(t, t1).item()],
                    [lbd(log0, t1).item(), lbd(log1, t1).item(), lbd(logt, t1).item()]])
    print(at1)


    # %% Weight init experimenting
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    # Apply the weights_init function to all layers of the model
    print(next(iter(model.parameters())).mean())
    model.apply(weights_init)
    print(next(iter(model.parameters())).mean())

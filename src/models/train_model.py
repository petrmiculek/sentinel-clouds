""" Train a model. """
import sys
# remove last element from path
import os
from os.path import abspath, join, exists
sys.path.pop()  # preexisting imports path messing up imports
sys.path.append(abspath(join('..')))  # ,'src'
# print("Python imports path:", "\n".join(sys.path))
# %%
# standard library
from copy import deepcopy
from types import SimpleNamespace
# external
import numpy as np
import torch
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import sigmoid
from torch.cuda.amp import GradScaler
from torch import autocast, sigmoid
from tqdm.auto import tqdm
from torchinfo import summary
import wandb as wb
# local
from src.data.dataset import get_loaders, get_loaders_processed
from src.models.unet import UNet
from src.models.util import EarlyStopping, evaluate_metrics, compute_metrics_own, keys_append, DiceLoss

# set numpy/torch print precision .4f
np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)
# %%
# Hyperparameters
HP = SimpleNamespace()
''' Preprocessing '''
HP.tile_size = 224
HP.crop_pad_mask = 'crop'
# -
''' Data '''
HP.workers = 0
HP.batch_size = 4
''' Model '''
# -
''' Training '''
HP.epochs = 1
# HP.warmup_prop = 0.1  # TODO lr warmup
HP.lr = 1e-3
wb.init(project="clouds", config=HP)  # , mode='disabled' if args.no_log else None)
run_name = wb.run.name
outputs_dir = join('runs', run_name)
os.makedirs(outputs_dir, exist_ok=True)
checkpoint_path = join(outputs_dir, 'model_checkpoint.pt')
# %% 
path_data = '/mnt/sdb1/code/sentinel2/processed'
# path_data = '/storage/brno2/home/petrmiculek/sentinel2/processed'
dataset_kwargs = {'tile_size': HP.tile_size, 'crop_pad_mask': HP.crop_pad_mask}  # doesn't matter when using processed data
loader_kwargs = {'batch_size': HP.batch_size, 'num_workers': HP.workers, 'pin_memory': True}
loader = get_loaders_processed(path_data, splits=['test', 'val'], **dataset_kwargs, **loader_kwargs)
loader['train'] = loader['test']  # TODO: remove
# %% Model + training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=4).to(device)
model_summary = summary(model, input_size=(1, 4, 224, 224))
# print(model_summary)  # already printed in the summary call
criterion = BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=HP.lr, weight_decay=1e-2)
scaler = GradScaler()  # mixed precision training (16-bit)
early_stopping = EarlyStopping(patience=5, path=checkpoint_path)
scheduler = ReduceLROnPlateau(optimizer, patience=3)
# wb.watch(model, log='all', log_freq=100)
wb.config.update({'criterion': criterion.__class__.__name__,
                   'optimizer': optimizer.__class__.__name__,
                   'scheduler': scheduler.__class__.__name__,
                    'architecture': model.__class__.__name__,
                    'train_size': len(loader['train'].dataset),
                   'model_bytes': model_summary.total_param_bytes,
                   'model_params': model_summary.total_params})
# %% Training
best_accu_val = 0
best_res = None
epochs_trained = 0
stop_training = False
grad_acc_steps = 1
for epoch in range(epochs_trained, epochs_trained + HP.epochs):
    model.train()
    ep_train_loss = 0
    metrics_train = []
    try:
        progress_bar = tqdm(loader['train'], mininterval=1., desc=f'ep{epoch} train')
        for i, sample in enumerate(progress_bar, start=1):
            img, label = sample['image'].to(device, non_blocking=True), sample['label'].to(device, non_blocking=True)
            # forward pass
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(img)  # prediction
                loss = criterion(logits, label)
            # backward pass
            scaler.scale(loss).backward()
            if i % grad_acc_steps == 0:  # gradient step with accumulated gradients
                scaler.step(optimizer)
                scaler.update()
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
    metrics_train = {k: np.mean([m[k] for m in metrics_train]) for k in metrics_train[0].keys()}
    metrics_train = keys_append(metrics_train, ' Training')
    print(metrics_train)
    ''' Validation loop '''
    model.eval()
    metrics_val = evaluate_metrics(model, loader['val'], criterion, suffix=' Validation')
    # log results
    res_epoch = {'Loss Training': ep_train_loss, 'Loss Validation': metrics_val['Loss Validation'],
                 'Accuracy Training': metrics_train['Accuracy Training'],
                   'Accuracy Validation': metrics_val['Accuracy Validation']}
    print(f'Epoch {epoch} results: ', res_epoch)
    wb.log(res_epoch, step=epoch)
    # print_dict(res_epoch)
    if metrics_val['Accuracy Validation'] >= best_accu_val:  # save best results
        best_accu_val = metrics_val['Accuracy Validation']
        # save a deepcopy of res to best_res
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
    img_i = img[i].cpu().numpy().transpose((1, 2, 0))
    label_i = np.stack([label[i].cpu().numpy()] * 3, axis=-1)
    pred_i = np.stack([pred[i].cpu().numpy()] * 3, axis=-1)
    table.add_data(wb.Image(img_i), wb.Image(label_i), wb.Image(pred_i))

wb.log({'Training Batch': table})
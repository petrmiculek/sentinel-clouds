# Cloud Segmentation
Author: Petr Mičulek
## Introduction
This report describes training a model for segmenting clouds in satellite data.

## Related work

Some commonly used semantic segmentation model architectures include UNet, UNet++ and DeepLabV3.
These are all encoder-decoder architectures, commonly using a backbone classifier network as the encoder.
The input-output dimensions tend to be the same, and there can be an additional classification task used on top of the segmentation itself.

For this task, the model can be evaluated with the following metrics. All of them are considered binary (no_cloud/cloud), and build on the standard metrics like TP (True Positive), TN, etc. 
- Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $\frac{TP}{TP + FP}$
- Recall: $\frac{TP}{TP + FN}$
- F-1 Score: $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- Dice Score: $\frac{2 \cdot TP}{2\cdot TP + FP + FN}$
- Intersection over Union (IoU, Jaccard Score): $\frac{TP}{TP + FP + FN}$
- Mathews Correlation Coefficient (MCC): $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP) \cdot (TP + FN) \cdot (TN + FP) \cdot (TN + FN)}}$

Loss functions for training the model can be the standard (Binary)Cross-Entropy (BCE), Mean Square Error, or any of the aforementioned metrics (adapted to a "lower is better" version). Focal loss, which is basically differently weighted BCE, can also be used.

## Data and preprocessing
The [Sentinel-2 dataset](https://zenodo.org/records/4172871) is split into training/validation/test splits with a $80:10:10$ ratio. The splitting randomly divides the dataset samples filenames, and saves only their filelists to keep the splitting details separate from source data organisation. 

This initial version of the dataset (CloudRawDataset) works with raw extracted dataset samples, which ends up repeatedly preprocessing them and leads to slow loading times. For this reason, a preprocessed version of the dataset is created and later used for all experiments (CloudProcessedDataset).

The preprocessed version takes a fixed approach to the dataset split and preprocessing, and saves only a single compressed .npz file for each dataset split. 

Firstly, the preprocessing drops unused channels of the data samples, so that only the image RGB + NIR channels and label masks are kept. Secondly, original data samples are about 1000x1000 pixels large, so these are cut into tiles of the desired native model size (224x224). Two tiles from the same image always end up in the same dataset split, so as to prevent training-test contamination. The splitting is otherwise random.

Since the image size does not divide the tile size, the incomplete tile data is cropped. The implementation also offers padding the image, and creating an additional mask to mark the valid image area. Padding and masking are not used in the final model.

```
Raw dataset directory structure:
<dataset_root>
    - filelists
        - train/val/test.csv
    - scenes
        - <file1..N>
    - masks
        - <file1..N>

Preprocessed dataset directory structure:
<dataset_root>
    - train.npz
    - val.npz
    - test.npz
```

## Design and implementation
The solution contains the following scripts/notebooks:
- data exploration 
- dataset splitting notebook
- dataset preprocessing script
- training script
- evaluation notebook

As per standard procedure, the model is trained on the training dataset split and evaluated every epoch on the validation split. The best-performing model is saved, and then finally evaluated on the test set, which is only touched once. 
The model output consists of logits, which are passed through a sigmoid function to produce the probability-like predictions. These predictions are then thresholded at 0.5 to obtain the final binary segmentation.

## Experiments
All training runs are logged in the [Weights\&Biases workspace](https://wandb.ai/petrmiculek/clouds). Further descriptions comment on what can be found in the workspace.
Initially, I trained only on 10\% of the dataset to iterate more quickly while finding suitable hyperparameters (train_size=832). The hyperparameters considered were: model architecture, model encoder backbone architecture, (multi-)loss choice, loss weighing, optimizer, learning rate, and learning rate scheduling. I manually tweaked those parameters on successive model runs to gain understanding of their behaviour.

Except for MCC, all the metrics mentioned before were used to monitor training progress.

Failing to train own UNet model implementation, I switched to an external library[^1] for the model architecture and backbone. Using the Binary Cross-Entropy as a loss function did not bring any results at first, so other loss functions were explored. 
The first successful hyperparameters were the following (respectively):
UNet, Resnet18, BinaryCrossEntropy loss, no weighing, Adam, 1e-4, no scheduling.

Upon the first success, more models were trained using various loss combinations (e.g., 1\*BCE + 0.1\*Dice), while keeping the previously mentioned hyperparameters fixed. The optimal loss configuration turned out to be 1\*BCE + 1\* MCC. In general, optimizing for accuracy (through BCE) leads to a model better performing even on the other metrics. On the contrary, the segmentation-specific losses do not create a strong model on their own. Their gradients often end up unstable. This instability also made 16-bit training impossible.

At the same time, learning rate scheduling has improved the training process further. A Cosine learning rate scheduler, warming up over 5 epochs from 1e-5 to 1e-4, and then decaying to 2e-8 over the rest of the 80 training epochs.

Using a larger encoder backbone (e.g. ResNet 152, RegNetY320) has not shown any benefit, but no further hyperparameter tuning has been attempted for these.

The best model's results (hyperparameters described above) are shown in the table below.
| Metric    	| Validation Set 	| Test Set 	|
|-----------	|----------------:  |----------:|
| Accuracy  	|         0.9301 	| 0.9196  	|
| Precision 	|         0.6019 	| 0.5678   	|
| Recall    	|         0.5604 	| 0.5355   	|
| Dice      	|         0.5617 	| 0.5348   	|
| F-1       	|         0.5617 	| 0.5348   	|
| IoU       	|         0.5364 	| 0.5168   	|

The evalution above presents the results computed within PyTorch. Sadly, when exporting the model to ONNX, the behaviour seems to change for the worse. I suspect the strided operations to be the cause of the error, but I did not manage to resolve this.

[^1]: [Segmentation Models Pytorch](https://smp.readthedocs.io)

## Conclusion
A UNet model with a Resnet18 backbone encoder was trained on the Sentinel-2 dataset, and reached an IoU of 0.51 on the test set. Sadly, the exported ONNX model performs worse, despite the same exact configuration otherwise.

Possible future work includes, in the decreasing order of relevance:
- fixing ONNX strided operations
- using padded images (and masking the loss)
- augmentations
- quantizing/pruning the model
- larger model backbone
- Fixed seed for reproducibility
- W\&B Artifacts
- weighing positive/negative samples for BCE

## Miscellaneous
- When re-building the original image from individual predictions, artifacts appear at the tile borders.
- Evaluation functions are averaged over a single batch, and then these results are again averaged. Given the different size of the last batch, this leads to inconsistent evaluation results across different batch sizes. For this reason, a batch size of 1 is used for the validation and test dataset splits.


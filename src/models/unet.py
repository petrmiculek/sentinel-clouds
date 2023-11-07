"""
    :filename UNet.py

    :brief Modified U-Net architecture.

    The original U-Net paper: https://arxiv.org/abs/1505.04597.

    This implementation contains modified U-Net architecture.
    The modification is in terms of added Batch Normalisation layer between Convolution and ReLU layers.
    Another modification is the possibility of the usage of upsampling in the decoder part.
    This implementation also uses 1:1 input output dimension sizes, i.e., if an input has dimensions of c x 128 x 128,
    the output is o_x x 128 x 128.

    :author Tibor Kubik
    :author Petr Miculek

    :email xkubik34@stud.fit.vutbr.cz
    :email xmicul08@stud.fit.vutbr.cz

    File was created as a part of project 'Image super-resolution for rendered volumetric data' for POVa/2021Z course.
"""

import torch
import torchvision.transforms.functional as TF

from torch import nn
from prettytable import PrettyTable


class DoubleConv(nn.Module):
    """Double convolution: Conv -> ReLU -> Conv -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBatchNorm(nn.Module):
    """Double convolution with additional BN: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBatchNorm, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upconvolution with added BN."""

    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """
    Modified U-Net architecture.
    What can be specified:

    in_channels:    The number of input feature channels. In this work is used one input channel.
    out_channels:   The number of output channels. It should correspond with the number of output segmented classes.
                    In case of the landmark regression, it should correspond with the number of detected landmarks
                    in the picture.
    batch_norm:     This flag specifies whether to add addition BN layers. Recommended value is True, as it helps
                    during the training.
    decoder_mode:   It specifies whether the Transposed Convolution layers should be used in the decoder part or the
                    Bilinear Upsampling.
    """

    def __init__(self,
                 in_channels=1,
                 batch_norm=True,
                 decoder_mode='upconv'):
        super(UNet, self).__init__()
        out_channels = 1  # note: not an argument
        assert (decoder_mode == 'upconv' or decoder_mode == 'upsample')

        self.batch_norm = batch_norm
        self.decoder_mode = decoder_mode

        features = [32, 64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        for feature in features:
            if self.batch_norm:
                self.downs.append(
                    DoubleConvBatchNorm(in_channels=in_channels,
                                        out_channels=feature
                                        )
                )
            else:
                self.downs.append(
                    DoubleConv(in_channels=in_channels,
                               out_channels=feature
                               )
                )
            in_channels = feature

        # Decoder part
        for feature in reversed(features):
            if self.decoder_mode == 'upconv':
                self.ups.append(
                    nn.ConvTranspose2d(
                        in_channels=feature * 2,
                        out_channels=feature,
                        kernel_size=2,
                        stride=2
                    )
                )
            else:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(mode='bilinear', scale_factor=2),
                        nn.Conv2d(feature * 2, feature, kernel_size=1),
                    )
                )

            if self.batch_norm:
                self.ups.append(
                    DoubleConvBatchNorm(
                        in_channels=feature * 2,
                        out_channels=feature
                    )
                )
            else:
                self.ups.append(
                    DoubleConv(
                        in_channels=feature * 2,
                        out_channels=feature
                    )
                )

        if self.batch_norm:
            self.bottleneck = DoubleConvBatchNorm(
                in_channels=features[-1],
                out_channels=features[-1] * 2,
            )
        else:
            self.bottleneck = DoubleConv(
                in_channels=features[-1],
                out_channels=features[-1] * 2,
            )

        self.final = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,  # the expected number of output channels, 1 for each landmark heatmap
            kernel_size=1  # the w and h of image would not be changed
        )
        # softmax over channels - unused
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips_conns = []  # For skip connections storing, typical for Unet-like architectures

        for down_step in self.downs:
            x = down_step(x)
            skips_conns.append(x)
            x = self.max_pool(x)

        x = self.bottleneck(x)
        skips_conns = skips_conns[::-1]  # Reverse the skip conns list to access proper element

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skips_conn = skips_conns[idx // 2]

            if x.shape != skips_conn.shape:
                x = TF.resize(x, size=skips_conn.shape[2:])  # If they don't match before concatenating, reshape

            concat = torch.cat((skips_conn, x), dim=1)

            x = self.ups[idx + 1](concat)

        x = self.final(x)
        return x  # logits
    
    def predict(self, x):
        return self.sigmoid(self.forward(x))  # probabilities

def count_parameters(model):
    """
    Counts total number of trainable parameters of given torch model. Prints table of its layers.

    :param model: torch model of NN

    :return: Number of trainable parameters.
    """

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params

if __name__ == '__main__':
    # In channels: 3x RGB, 1x NIR -> 4ch
    model = UNet(in_channels=4).to('cuda:0')

    input_tensor = torch.rand(1, 4, 224, 224).to('cuda:0')
    output_tensor = model(input_tensor)
    print(f'Output tensor size: {output_tensor.shape}.')


    # model_name = model.__class__.__name__
    # from torch.onnx import export
    # export_output = export(model, input_tensor, f"{model_name}.onnx")

    # from torchvision.models.segmentation import lraspp

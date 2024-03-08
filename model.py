import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.elu3(y)
        return y


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_levels):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_levels):
            dilation_size = 2**i
            self.layers.append(
                DilatedConv(
                    in_channels, out_channels, kernel_size, dilation=dilation_size
                )
            )

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


class BeatTracker(nn.Module):
    def __init__(self):
        super(BeatTracker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.debug = False
        self.melspec = MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=81,
        ).to(self.device)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            padding=((3 - 1) // 2, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            padding=((3 - 1) // 2, 0),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 8),
            padding=((1 - 1) // 2, 0),
        )
        self.pool1 = nn.MaxPool2d((1, 3))
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout = nn.Dropout(0.1)
        self.tcn = TCN(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            num_levels=11,
        )
        self.out = nn.Conv1d(16, 1, 1)

        self.to(self.device)

    def forward(self, x):
        # add batch if not present
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        # y = torch.log(self.melspec(x) + 1e-8)
        y = self.melspec(x)
        if self.debug:
            for i, item in enumerate(y):
                plt.figure()
                plt.imshow(item[0].detach().cpu().numpy()[:, :1000])
                plt.show()
                plt.savefig(f"model_outputs/mel_{i}.png")
        y = y.permute(0, 1, 3, 2)  # permute to (batch, channels, time, freq)
        # y = self.dropout(self.pool1(F.elu(self.conv1(y))))
        y = self.pool1(F.elu(self.conv1(y)))
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.imshow(item[0].detach().cpu().numpy()[:, :1000], aspect="auto")
                plt.show()
                plt.savefig(f"model_outputs/conv1_{i}.png")
        # y = self.dropout(self.pool2(F.elu(self.conv2(y))))
        y = self.pool2(F.elu(self.conv2(y)))
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.imshow(item[0].detach().cpu().numpy()[:, :1000], aspect="auto")
                plt.show()
                plt.savefig(f"model_outputs/conv2_{i}.png")
        # y = self.dropout(F.elu(self.conv3(y)))
        y = F.elu(self.conv3(y))
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.imshow(item[0].detach().cpu().numpy()[:, :1000], aspect="auto")
                plt.show()
                plt.savefig(f"model_outputs/conv3_{i}.png")
        y = y.squeeze(3)  # squeeze "summarized" frequency dim, (batch, channels, time)
        y = self.tcn(y)
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.imshow(item.detach().cpu().numpy()[:, :1000], aspect="auto")
                plt.show()
                plt.savefig(f"model_outputs/tcn_{i}.png")
        y = self.out(y).squeeze(1)
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.plot(item.detach().cpu().numpy()[:1000])
                plt.show()
                plt.savefig(f"model_outputs/activ_{i}.png")
        y = torch.sigmoid(y)
        if self.debug:
            for i, item in enumerate(y):
                plt.figure(figsize=(10, 10))
                plt.plot(item.detach().cpu().numpy()[:1000])
                plt.show()
                plt.savefig(f"model_outputs/sigmoid_{i}.png")
        return y


import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TCNLayer(nn.Module):

    def __init__(
        self, inputs, outputs, dilation, kernel_size=5, stride=1, padding=4, dropout=0.1
    ):

        super(TCNLayer, self).__init__()

        self.conv1 = nn.Conv1d(
            inputs,
            outputs,
            kernel_size,
            stride=stride,
            padding=int(padding / 2),
            dilation=dilation,
        )
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            inputs,
            outputs,
            kernel_size,
            stride=stride,
            padding=int(padding / 2),
            dilation=dilation,
        )
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.elu3 = nn.ELU()

    def forward(self, x):

        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.elu3(y)

        return y


class fullTCN(nn.Module):

    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):

        super(fullTCN, self).__init__()

        self.layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2**i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.layers.append(
                TCNLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):

        y = self.net(x)
        return y


class BeatTrackingTCN(nn.Module):

    def __init__(self, channels=16, kernel_size=5, dropout=0.1):

        super(BeatTrackingTCN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, channels, (3, 3), padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 3)),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), padding=(1, 0)),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 3)),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 8)), nn.ELU(), nn.Dropout(dropout)
        )

        self.frontend = nn.Sequential(self.convblock1, self.convblock2, self.convblock3)

        self.tcn = fullTCN(channels, [channels] * 11, kernel_size, dropout)

        self.out = nn.Conv1d(channels, 1, 1)

        self.to(self.device)

    def forward(self, spec):

        # if not batched
        if len(spec.shape) != 4:
            spec = spec.unsqueeze(0)
        spec = spec.permute(0, 1, 3, 2)  # (batch, channels, time, freq)

        frontend = self.frontend(spec)

        pre_tcn = frontend.squeeze(-1)
        tcn_out = self.tcn(pre_tcn)

        logits = self.out(tcn_out)

        return torch.sigmoid(logits).squeeze(1)

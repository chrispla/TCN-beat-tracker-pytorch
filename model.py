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
            dilation=2,
            # padding=(kernel_size - 1) * dilation,
        )
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=2,
            # padding=(kernel_size - 1) * dilation,
        )
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
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

        self.melspec = MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=int(np.floor(0.01 * 44100)),
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=81,
        ).to(self.device)
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.conv2 = nn.Conv2d(16, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 16, (1, 8))
        self.pool1 = nn.MaxPool2d((1, 3))
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout = nn.Dropout(0.1)
        self.tcn = TCN(16, 16, 5, 11)
        self.out = nn.Conv1d(16, 1, 5)

        self.to(self.device)

    def forward(self, x):
        y = self.melspec(x)
        y = y.permute(0, 1, 3, 2)  # permute to (batch, channels, time, freq)
        y = self.dropout(self.pool1(F.elu(self.conv1(y))))
        y = self.dropout(self.pool2(F.elu(self.conv2(y))))
        y = self.dropout(F.elu(self.conv3(y)))
        # we've "summarized" the frequency dimension, so we can squeeze it out
        # ending up with with (batch, channels, time)
        y = y.squeeze(3)
        y = self.tcn(y)
        y = self.out(y)
        y = torch.sigmoid(y)
        return y

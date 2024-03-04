import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )

    def forward(self, x):
        return self.conv(x)


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_levels):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_levels):
            dilation_size = 2**i
            self.layers.append(
                DilatedConv1d(
                    in_channels, out_channels, kernel_size, dilation=dilation_size
                )
            )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in self.layers:
            x = F.elu(self.dropout(layer(x)))
        return x


class BeatTracker(nn.Module):
    def __init__(self):
        super(BeatTracker, self).__init__()
        self.melspec = MelSpectrogram(
            sample_rate=22050,
            n_fft=2048,
            win_length=2048,
            hop_length=int(np.floor(0.01 * 22050)),
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=96,
            htk=False,
        )
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.conv2 = nn.Conv2d(16, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 16, (1, 8))
        self.pool = nn.MaxPool2d((1, 3), (1, 3))
        self.dropout = nn.Dropout(0.1)
        self.tcn = TCN(16, 16, 5, 20)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.melspec(x)
        x = x.unsqueeze(1)
        x = self.pool(F.elu(self.dropout(self.conv1(x))))
        x = self.pool(F.elu(self.dropout(self.conv2(x))))
        x = F.elu(self.dropout(self.conv3(x)))
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = torch.sigmoid(self.fc(x))
        return x

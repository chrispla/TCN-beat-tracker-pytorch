import pathlib

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class Ballroom(Dataset):
    def __init__(self, audio_dir, annotation_dir):

        # audio params from paper
        self.sr = 22050
        self.hop = int(np.floor(0.01 * self.sr))

        self.audio_dir = audio_dir

        # open file with list of audio files
        with open("allBallroomFiles", "r") as f:
            self.audio_files = f.readlines()

        self.basenames = [f.strip() for f in self.audio_files]

        # add audio dir to each file with pathlib
        self.audio_files = [
            pathlib.Path(audio_dir) / f.strip() for f in self.audio_files
        ]

        # load annotations
        self.annotation_dir = annotation_dir
        self.beat_times = {}
        self.downbeat_times = {}

        for basename in self.basenames:
            annotation_path = pathlib.Path(annotation_dir) / (basename + ".beats")
            beats = []
            downbeats = []
            with open(annotation_path, "r") as f:
                raw_lines = f.readlines()
                for line in raw_lines:
                    beats.append(float(line.strip().split()[0]))
                    if line.strip().split()[1] == "1":
                        downbeats.append(float(line.strip().split()[0]))

            self.beat_times[basename] = beats
            self.downbeat_times[basename] = downbeats

        # compute beat and downbeat waveform frames given sr
        self.beat_wave_frames = {}
        self.downbeat_wave_frames = {}
        for k, v in self.beat_times.items():
            self.beat_wave_frames[k] = int(v * self.sr)
        for k, v in self.downbeat_times.items():
            self.downbeat_wave_frames[k] = int(v * self.sr)

        # compute beat and downbeat frames based on the sr and hop for the spectrogram
        self.beat_spec_frames = {}
        self.downbeat_spec_frames = {}
        for k, v in self.beat_times.items():
            self.beat_spec_frames[k] = int(v * self.sr / self.hop)
        for k, v in self.downbeat_times.items():
            self.downbeat_spec_frames[k] = int(v * self.sr / self.hop)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        basename = self.basenames[idx]

        # load audio
        audio, _ = torchaudio.load(audio_file, normalize=True)

        # load annotations
        beat_times = torch.tensor(self.beat_times[basename])
        downbeat_times = torch.tensor(self.downbeat_times[basename])

        return audio, beat_times, downbeat_times

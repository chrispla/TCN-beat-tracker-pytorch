from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class Ballroom(Dataset):
    def __init__(self, audio_dir, annotation_dir):

        # we're going to trim everything to this for simplicity
        self.min_duration = 29.35

        # audio params from paper
        self.sr = 44100
        self.hop = 441

        self.audio_dir = audio_dir

        # open file with list of audio files
        with open(Path(self.audio_dir) / "allBallroomFiles", "r") as f:
            self.audio_files = f.readlines()

        # get basename without extension
        self.basenames = [Path(f.strip()).stem for f in self.audio_files]

        # add audio dir to each file with pathlib
        self.audio_files = [Path(audio_dir) / f.strip() for f in self.audio_files]

        # load annotations
        self.annotation_dir = annotation_dir
        self.beat_times = {}
        self.downbeat_times = {}

        for basename in self.basenames:
            annotation_path = Path(annotation_dir) / (basename + ".beats")
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
            self.beat_wave_frames[k] = librosa.time_to_samples(v, sr=self.sr)
        for k, v in self.downbeat_times.items():
            self.downbeat_wave_frames[k] = librosa.time_to_samples(v, sr=self.sr)

        # compute beat and downbeat frames based on the sr and hop for the spectrogram
        self.beat_spec_frames = {}
        self.downbeat_spec_frames = {}
        for k, v in self.beat_times.items():
            self.beat_spec_frames[k] = librosa.time_to_frames(
                v, sr=self.sr, hop_length=self.hop, n_fft=2048
            )
        for k, v in self.downbeat_times.items():
            self.downbeat_spec_frames[k] = librosa.time_to_frames(
                v, sr=self.sr, hop_length=self.hop, n_fft=2048
            )

        # get beat vectors aligned with spectrogram frames
        # for training, they use 0.5 in a radius of 2 around frame, and 1 for the frame
        self.n_spec_frames = int(self.min_duration * self.sr / self.hop) + 1
        self.beat_vectors = {}
        self.downbeat_vectors = {}
        for k, v in self.beat_spec_frames.items():
            beat_vector = np.zeros(self.n_spec_frames)
            for frame in v:
                # generate indices around frame, considering edges
                indices = np.arange(
                    max(0, frame - 2), min(self.n_spec_frames, frame + 3)
                )
                # set frame to 1, rest 0.5s
                values = np.where(indices == frame, 1.0, 0.5)
                # set values in beat vector
                beat_vector[indices] = values
            self.beat_vectors[k] = beat_vector
        for k, v in self.downbeat_spec_frames.items():
            downbeat_vector = np.zeros(self.n_spec_frames)
            for frame in v:
                indices = np.arange(
                    max(0, frame - 2), min(self.n_spec_frames, frame + 3)
                )
                values = np.where(indices == frame, 1.0, 0.5)
                downbeat_vector[indices] = values
            self.downbeat_vectors[k] = downbeat_vector

        # get beat vectors aligned with waveform frames
        # for training, they use 0.5 in a radius of 2 around frame, and 1 for the frame
        self.beat_vectors_wave = {}
        self.downbeat_vectors_wave = {}
        for k, v in self.beat_wave_frames.items():
            beat_vector = np.zeros(v[-1] + 1)
            for frame in v:
                indices = np.arange(
                    max(0, frame - 2), min(beat_vector.shape[0], frame + 3)
                )
                values = np.where(indices == frame, 1.0, 0.5)
                beat_vector[indices] = values
            self.beat_vectors_wave[k] = beat_vector
        for k, v in self.downbeat_wave_frames.items():
            downbeat_vector = np.zeros(v[-1] + 1)
            for frame in v:
                indices = np.arange(
                    max(0, frame - 2), min(downbeat_vector.shape[0], frame + 3)
                )
                values = np.where(indices == frame, 1.0, 0.5)
                downbeat_vector[indices] = values
            self.downbeat_vectors_wave[k] = downbeat_vector

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        basename = self.basenames[idx]

        # load audio
        audio, og_sr = torchaudio.load(audio_file, normalize=True)
        # to mono, if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # resample to 44100
        audio = torchaudio.transforms.Resample(og_sr, 44100)(audio)

        # trim to min_duration
        audio = audio[:, : int(self.min_duration * self.sr)]

        # load annotations -- we're returning the train vector, so we
        # need to clean that up later for validation and test
        beat_vector = torch.tensor(self.beat_vectors[basename]).float()
        downbeat_vector = torch.tensor(self.downbeat_vectors[basename]).float()

        return (audio, beat_vector, downbeat_vector)

    def get_dataloaders(self, batch_size=1, num_workers=4):
        total_size = len(self)
        train_size = int(total_size * 0.1)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # simple test
    dataset = Ballroom(
        audio_dir="data/ballroom/audio", annotation_dir="data/ballroom/annotations"
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    for i, data in enumerate(train_loader, 0):
        inputs, beat_vector, downbeat_vector = data
        print(
            "Train shapes (audio, beat, downbeat):",
            inputs.shape,
            beat_vector.shape,
            downbeat_vector.shape,
        )
        break
    for i, data in enumerate(val_loader, 0):
        inputs, beat_vector, downbeat_vector = data
        print(
            "Val shapes (audio, beat, downbeat):",
            inputs.shape,
            beat_vector.shape,
            downbeat_vector.shape,
        )
        break
    for i, data in enumerate(test_loader, 0):
        inputs, beat_vector, downbeat_vector = data
        print(
            "Test shapes (audio, beat, downbeat):",
            inputs.shape,
            beat_vector.shape,
            downbeat_vector.shape,
        )
        break
    # print number of items in loaders
    print(
        "Lenghts (train, val, test):",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

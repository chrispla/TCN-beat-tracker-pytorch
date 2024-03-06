"""Compute beats and downbeats from an audio file."""

import torch
import torchaudio

from model import BeatTracker
from utils import output_to_beat_times


def beatTracker(inputFile):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load audio
    audio, og_sr = torchaudio.load(inputFile, normalize=True)
    # to mono, if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # resample to 44100
    audio = torchaudio.transforms.Resample(og_sr, 44100)(audio)
    audio = audio.to(device)

    # compute beats
    beat_model = BeatTracker()
    beat_model.load_state_dict(torch.load("models/beat_tracker_1.13.pt"))
    beat_model.eval()
    output = beat_model(audio)
    beat_times = output_to_beat_times(
        output.squeeze().detach().cpu().numpy(),
        sr=44100,
        hop=0.01 * 44100,
        model_type="beats",
    )

    # compute downbeats
    downbeat_model = BeatTracker()
    downbeat_model.load_state_dict(torch.load("models/downbeat_tracker_1.13.pt"))
    downbeat_model.eval()
    output = downbeat_model(audio)
    downbeat_times = output_to_beat_times(
        output.squeeze().detach().cpu().numpy(),
        sr=44100,
        hop=0.01 * 44100,
        model_type="downbeats",
    )

    return beat_times, downbeat_times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute beats and downbeats")
    parser.add_argument("-i", type=str, help="path to the input audio file")
    args = parser.parse_args()

    beat_times, downbeat_times = beatTracker(args.i)
    print("Beat times:", beat_times)
    print("Downbeat times:", downbeat_times)

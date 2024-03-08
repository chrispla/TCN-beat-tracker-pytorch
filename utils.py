"""Utilities for beat processing."""

import librosa
import numpy as np
from madmom.features import DBNBeatTrackingProcessor


def vector_to_times(beat_vector, sr, hop):
    """Convert beat vector to beat times."""
    frames = np.where(beat_vector == 1.0)[1]
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop, n_fft=2048)


def output_to_beat_times(output, sr, hop, model_type):
    """Convert model output to beat times using DBN."""
    if model_type == "beats":
        min_bpm, max_bpm = 55, 215
    elif model_type == "downbeats":
        min_bpm, max_bpm = 10, 75

    postprocessor = DBNBeatTrackingProcessor(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=sr / hop,
        online=False,
        transition_lambda=100,
    )
    return postprocessor.process_offline(output)

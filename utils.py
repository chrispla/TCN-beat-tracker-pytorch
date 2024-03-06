"""Utilities for beat processing."""

import numpy as np


def vector_to_times(beat_vector, sr, hop):
    """Convert beat vector to beat times."""
    # find frames with 1
    frames = np.where(beat_vector == 1.0)[1]
    # convert frames to times
    times = frames * hop / sr
    return times


def output_to_beat_times(output, sr, hop):
    """Convert model output to beat times using DBN."""
    pass

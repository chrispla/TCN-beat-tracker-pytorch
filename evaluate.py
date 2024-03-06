import argparse
from pathlib import Path

import mir_eval
import torch

from dataset import Ballroom
from model import BeatTracker
from utils import output_to_beat_times, vector_to_times

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--name", "-n", type=float, required=False, help="model name")
parser.add_argument(
    "--type", "-t", type=str, required=True, help="model type (beats or downbeats)"
)
args = parser.parse_args()
model_type = args.type

if not args.name:
    # get the model with the lowest validation loss (split("_")[-1][-3])
    if model_type == "beats":
        models = list(Path("models").glob("beat_tracker_*"))
    elif model_type == "downbeats":
        models = list(Path("models").glob("downbeat_tracker_*"))
    models.sort(key=lambda x: float(x.stem.split("_")[-1][-3:]))
    try:
        model_dir = models[0]
    except IndexError:
        print("No models found in ./models/")
        exit()
else:
    model_dir = Path("models", args.name)
print("Evaluating model:", model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = BeatTracker()
model.load_state_dict(torch.load(model_dir))

model.eval()
dataset = Ballroom(
    audio_dir="data/ballroom/audio", annotation_dir="data/ballroom/annotations"
)
_, _, test_loader = dataset.get_dataloaders(batch_size=1)

# evaluate
metrics = {"f1": [], "CMLc": [], "CMLt": [], "AMLc": [], "AMLt": [], "D": []}
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, beat_vector, downbeat_vector = data
        outputs = model(inputs.to(device))
        output_times = output_to_beat_times(
            outputs.squeeze().cpu().numpy(),
            sr=44100,
            hop=0.01 * 44100,
            model_type=model_type,
        )
        if model_type == "beats":
            target_times = vector_to_times(beat_vector, sr=44100, hop=0.01 * 44100)
        elif model_type == "downbeats":
            target_times = vector_to_times(downbeat_vector, sr=44100, hop=0.01 * 44100)

        # compute metrics
        metrics["f1"].append(mir_eval.beat.f_measure(target_times, output_times))
        CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(target_times, output_times)
        metrics["CMLc"].append(CMLc)
        metrics["CMLt"].append(CMLt)
        metrics["AMLc"].append(AMLc)
        metrics["AMLt"].append(AMLt)
        metrics["D"].append(mir_eval.beat.information_gain(target_times, output_times))

# average each list
for key in metrics:
    metrics[key] = sum(metrics[key]) / len(metrics[key])

print(
    "Metrics\n",
    f"F1  : {metrics['f1']:.3f}\n",
    f"CMLc: {metrics['CMLc']:.3f}\n",
    f"CMLt: {metrics['CMLt']:.3f}\n",
    f"AMLc: {metrics['AMLc']:.3f}\n",
    f"AMLt: {metrics['AMLt']:.3f}\n",
    f"D   : {metrics['D']:.3f}\n",
)

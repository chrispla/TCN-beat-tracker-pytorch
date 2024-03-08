import argparse
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

import utils
from dataset import Ballroom
from model import BeatTracker, BeatTrackingTCN

logging = True


parser = argparse.ArgumentParser(description="Train model")
parser.add_argument(
    "--model_type",
    "-t",
    type=str,
    default="beats",
    help="model type (beats or downbeats)",
)
args = parser.parse_args()


# torch.manual_seed(42)
model_type = args.model_type
# model = BeatTracker()
model = BeatTrackingTCN()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10, min_lr=1e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if logging is True:
    import wandb

    wandb.init(project="beat-tracking")
    wandb.watch(model, log="all")

model.train()
running_loss = 0.0

best_val_loss = float("inf")
epochs_no_improve = 0
n_epochs_stop = 500

# get dataset splits
dataset = Ballroom(
    audio_dir="data/ballroom/audio", annotation_dir="data/ballroom/annotations"
)
train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=16)

melspec = MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=441, n_mels=81).to(
    device
)

# Train
for epoch in range(500):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, data in pbar:
        inputs, beat_vector, downbeat_vector = data
        if model_type == "beats":
            labels = beat_vector
        elif model_type == "downbeats":
            labels = downbeat_vector
        else:
            raise ValueError("Invalid model type, needs to be 'beats' or 'downbeats'.")
        # plot beats and downbeats
        # plt.figure()
        # plt.plot(beat_vector.squeeze().cpu().numpy(), c="r")
        # plt.plot(downbeat_vector.squeeze().cpu().numpy(), c="b")
        # plt.show()
        # plt.savefig("model_outputs/groundtruth.png")

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(melspec(inputs))

        # # plot activations and ground truth
        # plt.figure()
        # output_local = outputs[0][:1000].squeeze().cpu().detach().numpy()
        # # normalize between 0 and 1
        # output_local = (output_local - output_local.min()) / (
        #     output_local.max() - output_local.min()
        # )
        # plt.plot(output_local, c="r")
        # # time.sleep(0.1)
        # plt.plot(labels[0][:1000].squeeze().cpu().detach().numpy(), c="b")
        # plt.show()
        # plt.savefig("model_outputs/predictions.png")

        # # plot mel spec and ground truth
        # melspec = MelSpectrogram(
        #     sample_rate=44100,
        #     n_fft=2048,
        #     win_length=2048,
        #     hop_length=441,
        #     center=True,
        #     pad_mode="reflect",
        #     power=2.0,
        #     norm="slaney",
        #     n_mels=81,
        # ).to(device)
        # mel = torch.log(melspec(inputs)[0] + 1e-8)

        # fix, ax = plt.subplots()
        # ax.imshow(mel[0].detach().cpu().numpy()[:, :1000])
        # ax.plot(labels[0][:1000].squeeze().cpu().detach().numpy() * 80, c="r")
        # plt.show()
        # plt.savefig("model_outputs/mel.png")

        # # plot audio and ground truth
        # plt.figure()
        # plt.plot(inputs[0][0][:].cpu().detach().numpy())
        # plt.plot(downbeat_vector_wave[0].detach().cpu().numpy(), c="r")
        # plt.show()
        # plt.savefig("model_outputs/audio.png")

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss/(i+1):.4f}")

        if logging is True:
            wandb.log({"Training loss": loss.item()})

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, beat_vector, downbeat_vector = data

            if model_type == "beats":
                labels = beat_vector
            elif model_type == "downbeats":
                labels = downbeat_vector

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(melspec(inputs))
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f},",
        f"Validation Loss: {val_loss/len(val_loader):.4f}",
    )

    scheduler.step(val_loss)

    if logging is True:
        wandb.log({"Validation loss": val_loss})
        wandb.log({"Learning Rate": optimizer.param_groups[0]["lr"]})

    # Check early stopping condition
    if val_loss < best_val_loss:
        epochs_no_improve = 0
        best_val_loss = val_loss
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print("Early stopping!")
            break

# save model
Path("models").mkdir(exist_ok=True)  # create model dir if it doesn't exist
if model_type == "beats":
    torch.save(model.state_dict(), Path("models") / f"beat_tracker_{val_loss:.3f}.pt")
elif model_type == "downbeats":
    torch.save(
        model.state_dict(), Path("models") / f"downbeat_tracker_{val_loss:.3f}.pt"
    )

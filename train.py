import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import Ballroom
from model import BeatTracker

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument(
    "--model_type",
    "-t",
    type=str,
    default="beats",
    help="model type (beats or downbeats)",
)
args = parser.parse_args()


torch.manual_seed(0)
model_type = args.model_type
model = BeatTracker()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()
running_loss = 0.0

best_val_loss = float("inf")
epochs_no_improve = 0
n_epochs_stop = 50

# get dataset splits
dataset = Ballroom(
    audio_dir="data/ballroom/audio", annotation_dir="data/ballroom/annotations"
)
train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=8)

# Train
for epoch in range(200):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, data in pbar:
        inputs, beat_vector, downbeat_vector = data
        if model_type == "beats":
            labels = beat_vector
        elif model_type == "downbeats":
            labels = downbeat_vector
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss/(i+1):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, beat_vector, _ = data

            if model_type == "beats":
                labels = beat_vector
            elif model_type == "downbeats":
                labels = downbeat_vector

            # we need to clean up the 0.5 in the vector
            # !!! need to handle this more nicely later
            labels[labels == 0.5] = 0.0

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}"
    )

    scheduler.step(val_loss)

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
    torch.save(model.state_dict(), Path("models") / f"beat_tracker_{val_loss:.2f}.pt")
elif model_type == "downbeats":
    torch.save(
        model.state_dict(), Path("models") / f"downbeat_tracker_{val_loss:.2f}.pt"
    )

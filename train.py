import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import Ballroom
from model import BeatTracker

torch.manual_seed(0)

model = BeatTracker()
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10, verbose=True)
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
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [80, 10, 10]
)
batch_size = 8
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# Train
for epoch in range(200):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, beat_vector, _ = data
        inputs, beat_vector = inputs.to(device), beat_vector.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, beat_vector)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, beat_vector, _ = data
            inputs, beat_vector = inputs.to(device), beat_vector.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, beat_vector)
            val_loss += loss.item()

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
torch.save(model.state_dict(), f"beat_tracker_{val_loss:.2f}.pt")

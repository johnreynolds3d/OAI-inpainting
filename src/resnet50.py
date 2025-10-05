# Data Handling
import os  # For handling file and directory paths

import matplotlib.pyplot as plt  # For plotting graphs, the training loss curve at the end per epoch
import pandas as pd  # For reading and manipulating CSV/tabular data

# PyTorch and Deep Learning
import torch  # PyTorch library for tensor operations and GPU support (im doing this on a laptop with my cpu)
from PIL import Image  # For loading and manipulating images
from sklearn.model_selection import (
    train_test_split,
)  # For splitting the dataset into train/val/test
from torch import nn  # For building and using neural network layers (Linear, Conv, etc)
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # For creating custom datasets and loading them in batches

# For pretrained models and transforms
from torchvision import (
    models,
    transforms,
)  # `transforms` for image preprocessing, `models` for ResNet-50

# Paths
# Define the path to the CSV file that contains BMD values and image filenames
# Example row in CSV: 0.951781914, 6.E.1_9000099_20090728_001.png
CSV_PATH = "data/data.csv"

# Define the folder where the image files are stored
# Each filename in the CSV refers to an image inside this folder
IMAGE_FOLDER = "data/images"


# Dataset class
# Define a custom PyTorch Dataset for loading images and BMD values
class BMDDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe  # Store the input DataFrame that contains BMD values and image filenames
        self.transform = transform  # Optional image transformations (e.g., resizing, converting to tensor)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the row at the given index
        row = self.dataframe.iloc[idx]

        # Load the image using the filename from the CSV and convert it to RGB
        # duplicates the single grayscale channel into three identical channels (R = G = B).
        # Pretrained ResNet-50 expects 3 x 224 x 224 inputs (RGB)
        # Feeding it 1 x 224 x 224 (grayscale) would raise an error
        # Converting grayscale to RGB by duplicating the channel is a common workaround
        image = Image.open(os.path.join(IMAGE_FOLDER, row[1])).convert("RGB")

        # Apply any transformations (resize and convert to tensor)
        # Images are all ready 224 x 224
        if self.transform:
            image = self.transform(image)

        # Get the BMD value from the first column and convert it to a PyTorch float tensor
        label = torch.tensor(float(row[0]), dtype=torch.float32)

        # Return the image and its corresponding label
        return image, label


# Load data
# Load the CSV file into a DataFrame
# Each row contains: [BMD value, image filename]
df = pd.read_csv(CSV_PATH, header=None)

# Split the data into 90% train+validation and 10% test
# We set a random seed (random_state=1) for reproducibility, different seed means different splits and data in differe
# sections
train_val, test = train_test_split(df, test_size=0.1, random_state=1)

# Further split the 90% train+val portion into:
#   80% training
#   10% validation
# So we split train_val using test_size=0.1111 because:
#   0.1 * 90% =..kinda 10% (roughly matching the test split)
train, val = train_test_split(train_val, test_size=0.1, random_state=1)


# Transforms
# Define a set of image transformations to apply to every image
transform = transforms.Compose(
    [
        # Resize the image to 224x224 pixels
        # This is the input size expected by ResNet-50
        transforms.Resize((224, 224)),
        # Convert the image to a PyTorch tensor
        # Also scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),
        # Normalize the tensor using the mean and std used during ImageNet pretraining
        # These values are channel-wise (for RGB): [mean_R, mean_G, mean_B], [std_R, std_G, std_B]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Dataloaders
# Create the training DataLoader
# Uses the training portion of the dataset
# Applies the defined transforms (make sures its the right size and pixel values are between 0 - 1)
# Loads data in batches of 32
# Shuffles the data each epoch to help the model generalise better
train_loader = DataLoader(BMDDataset(train, transform), batch_size=32, shuffle=True)

# Create the validation DataLoader
# Used to evaluate the model during training (after each epoch)
# No shuffling needed since we aren't training on this data
val_loader = DataLoader(BMDDataset(val, transform), batch_size=32)

# Create the test DataLoader
# Used for final evaluation after training is complete
# Also doesn't need shuffling
test_loader = DataLoader(BMDDataset(test, transform), batch_size=32)

# Load and modify ResNet-50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pre-trained ResNet-50 model from torchvision
# pretrained=True means we load weights that were trained on the ImageNet dataset
# This allows us to leverage learned features (like edges, textures, shapes) from general-purpose images
# We'll fine-tune the last layers to adapt the model to our specific task (predicting BMD from X-ray images)
# This also severly cuts down training times as we are not adjusting another billion params as they are frozen
model = models.resnet50(pretrained=True)


# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last conv block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace FC layer
# Replace the final fully connected (fc) layer
# By default, ResNet-50's final layer is:
#     nn.Linear(2048, 1000)
# This is designed for ImageNet classification with 1000 output classes.
# But in our case, we are solving a regression problem, not classification.
# We want the model to predict a single continuous value: the BMD (bone mineral density).
# So we replace the fc layer with:
#     nn.Linear(2048, 1)
# This means the model will output a single floating-point value per image.
model.fc = nn.Linear(model.fc.in_features, 1)

# Move the model to GPU if available (or CPU otherwise)
model.to(device)

# Training setup

# Define the loss function
# Since this is a regression task (predicting continuous BMD values),
# we use Mean Squared Error (MSE) loss (doesn't have to be), a standard choice for regression problems.
criterion = nn.MSELoss()

# Set up the optimiser, we use Adam for efficient gradient-based optimisation
# We only update the parameters that are trainable (those with requires_grad=True, defined earlier)
# This includes the last conv block (layer4) and the new final fully connected layer (fc)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,  # Learning rate,small step size to avoid large updates during fine-tuning
)

# Set the number of training epochs (how many times the model sees the entire training set)
num_epochs = 10

# Initialise a list to store training loss values over time
# This is useful for plotting and evaluating the model’s learning progress
train_losses = []

# Training loop
print("Starting training...")
# Loop over the dataset for a number of epochs (complete passes through the training set)
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    running_loss = 0.0  # Reset cumulative loss for this epoch

    # Loop over each batch in the training DataLoader
    for images, labels in train_loader:
        # Move images and labels to the GPU (if available)
        # unsqueeze(1) adds dimension to match model output shape: [batch_size, 1]
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        # Forward pass: compute model predictions
        outputs = model(images)

        # Compute the loss between predictions and actual BMD labels
        loss = criterion(outputs, labels)

        # Zero out any previous gradients before the backward pass
        optimizer.zero_grad()

        # Backward pass: compute gradients
        loss.backward()

        # Update the model's trainable parameters using the optimiser
        optimizer.step()

        # Accumulate the loss, scaled by the batch size (to average later)
        running_loss += loss.item() * images.size(0)

    # Compute average loss for the whole epoch
    epoch_loss = running_loss / len(train_loader.dataset)

    # Store the loss for later plotting
    train_losses.append(epoch_loss)

    # Print progress so we can see the model learning over time
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "resnet50_bmd_model.pth")

# Plot training loss
plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("resnet50_training_loss.png")

# --- Evaluate on Test Set ---
model.eval()  # Set model to evaluation mode (disables dropout, etc.)
predictions = []
actuals = []

with torch.no_grad():  # Disable gradient computation for evaluation (faster, uses less memory)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        outputs = model(images)

        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# Convert to flat lists
predictions = [float(p) for p in predictions]
actuals = [float(a) for a in actuals]

# Save predictions and actuals to CSV
import csv

with open("resnet50_test_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual_BMD", "Predicted_BMD"])
    writer.writerows(zip(actuals, predictions))

# --- Plot predictions vs. actuals ---
plt.figure()
plt.scatter(actuals, predictions, alpha=0.6)
plt.plot(
    [min(actuals), max(actuals)],
    [min(actuals), max(actuals)],
    color="red",
    linestyle="--",
    label="Ideal Fit",
)
plt.xlabel("Actual BMD")
plt.ylabel("Predicted BMD")
plt.title("ResNet-50 Predictions vs. Actual BMD (Test Set)")
plt.legend()
plt.grid(True)
plt.savefig("resnet50_test_scatter.png")
plt.show()

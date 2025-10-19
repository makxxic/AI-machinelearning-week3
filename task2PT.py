# ============================================================
# Task 2: Deep Learning with PyTorch
# Dataset: MNIST Handwritten Digits
# Goal: Build a CNN to classify digits and achieve >95% accuracy
# ============================================================

# Step 1: Import Libraries
import torch # Import the main PyTorch library
import torch.nn as nn # Import the neural network module
import torch.nn.functional as F # Import the functional module (contains activation functions, etc.)
import torch.optim as optim # Import the optimization module (contains optimizers like Adam)
from torchvision import datasets, transforms # Import datasets and transforms from torchvision (for common datasets and image transformations)
from torch.utils.data import DataLoader # Import DataLoader for efficient data loading in batches
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs
import numpy as np # Import numpy for numerical operations

# ------------------------------------------------------------
# Step 2: Define Transformations & Load Dataset
# ------------------------------------------------------------
# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values (mean and std for grayscale images)
])

# Download and load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Download and load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders for iterating through the datasets in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # DataLoader for training data, shuffled
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False) # DataLoader for test data, not shuffled

# ------------------------------------------------------------
# Step 3: Define the CNN Model Architecture
# ------------------------------------------------------------
# Define the CNN model as a PyTorch Module
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel size, 1 pixel padding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel size, 1 pixel padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer: 2x2 window size, stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # First fully connected layer: input size calculated from previous layers, 128 output units
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Second fully connected layer (output layer): 128 input units, 10 output units (for 10 classes)
        self.fc2 = nn.Linear(128, 10)

    # Define the forward pass of the model
    def forward(self, x):
        # Apply convolution 1, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply convolution 2, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Apply fully connected layer 1 and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply fully connected layer 2 (output layer)
        x = self.fc2(x)
        return x

# Instantiate the CNN model
model = CNNModel()

# ------------------------------------------------------------
# Step 4: Define Loss Function and Optimizer
# ------------------------------------------------------------
# Define the loss function: Cross Entropy Loss is suitable for multi-class classification
criterion = nn.CrossEntropyLoss()
# Define the optimizer: Adam optimizer with a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# Step 5: Train the Model
# ------------------------------------------------------------
# Define the number of training epochs
epochs = 5
# Lists to store training loss and accuracy for plotting
train_losses, train_accuracies = [], []

# Start the training loop
for epoch in range(epochs):
    running_loss = 0.0 # Initialize running loss for the epoch
    correct = 0 # Initialize correct predictions count
    total = 0 # Initialize total predictions count

    # Iterate over the training data in batches
    for images, labels in train_loader:
        # Zero the gradients of model parameters
        optimizer.zero_grad()

        # Perform forward pass: compute model outputs
        outputs = model(images)
        # Calculate the loss
        loss = criterion(outputs, labels)

        # Perform backward pass: compute gradients
        loss.backward()
        # Perform optimization step: update model weights
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

        # Calculate accuracy for the current batch
        _, predicted = torch.max(outputs.data, 1) # Get the predicted class (index with max probability)
        total += labels.size(0) # Add the batch size to the total count
        correct += (predicted == labels).sum().item() # Add the number of correct predictions to the correct count

    # Calculate average loss and accuracy for the epoch
    acc = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(acc)

    # Print training progress for the epoch
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {acc:.2f}%")

# Print a message indicating training is complete
print("\nTraining complete!")

# ------------------------------------------------------------
# Step 6: Evaluate on Test Data
# ------------------------------------------------------------
# Set the model to evaluation mode (disables dropout, etc.)
model.eval()
correct = 0 # Initialize correct predictions count for testing
total = 0 # Initialize total predictions count for testing
# Disable gradient calculation for evaluation
with torch.no_grad():
    # Iterate over the test data in batches
    for images, labels in test_loader:
        # Perform forward pass
        outputs = model(images)
        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Add the batch size to the total count
        total += labels.size(0)
        # Add the number of correct predictions to the correct count
        correct += (predicted == labels).sum().item()

# Calculate the test accuracy
test_accuracy = 100 * correct / total
# Print the test accuracy
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# ------------------------------------------------------------
# Step 7: Plot Training Accuracy Curve
# ------------------------------------------------------------
# Create a figure for the plot
plt.figure(figsize=(8, 5))
# Plot the training accuracies over epochs
plt.plot(range(1, epochs + 1), train_accuracies, marker='o')
# Set the title of the plot
plt.title('Training Accuracy over Epochs')
# Set the x-axis label
plt.xlabel('Epoch')
# Set the y-axis label
plt.ylabel('Accuracy (%)')
# Add a grid to the plot
plt.grid(True)
# Display the plot
plt.show()

# ------------------------------------------------------------
# Step 8: Visualize Model Predictions on Sample Images
# ------------------------------------------------------------
# Get a batch of test images from the test loader
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Predict the outputs for the sample images
outputs = model(images)
# Get the predicted classes
_, preds = torch.max(outputs, 1)

# Display a specified number of random sample images with predictions
num_samples = 5 # Define the number of samples to visualize
indices = np.random.choice(range(len(images)), num_samples, replace=False) # Select random indices

# Create a figure for the plot
plt.figure(figsize=(10, 4))
# Loop through the selected sample images
for i, idx in enumerate(indices):
    # Get the image and remove the channel dimension for plotting
    img = images[idx].numpy().squeeze()
    # Create a subplot for each image
    plt.subplot(1, num_samples, i + 1)
    # Display the image in grayscale
    plt.imshow(img, cmap='gray')
    # Set the title with predicted and true labels
    plt.title(f"Pred: {preds[idx].item()}\nTrue: {labels[idx].item()}")
    # Turn off the axes
    plt.axis('off')
# Show the plot
plt.show()
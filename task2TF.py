# ============================================================
# Task 2: Deep Learning with TensorFlow
# Dataset: MNIST Handwritten Digits
# Goal: Build a CNN to classify digits and achieve >95% accuracy
# ============================================================

# Import required libraries
import tensorflow as tf # Import TensorFlow, the deep learning framework
from tensorflow.keras import datasets, layers, models # Import specific modules for datasets, layers, and models from Keras (TensorFlow's high-level API)
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs
import numpy as np # Import numpy for numerical operations

# ------------------------------------------------------------
# Step 1: Load and preprocess the dataset
# ------------------------------------------------------------
# The MNIST dataset contains 60,000 training images and 10,000 test images
# Each image is 28x28 pixels (grayscale)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() # Load the MNIST dataset, splitting it into training and testing sets

# Normalize pixel values to the range [0,1] for better training stability
x_train, x_test = x_train / 255.0, x_test / 255.0 # Divide pixel values by 255 to scale them between 0 and 1

# Reshape data to include a channel dimension (needed for CNN input)
# Shape: (num_samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1) # Reshape training data to add a channel dimension (1 for grayscale)
x_test = x_test.reshape(-1, 28, 28, 1) # Reshape testing data to add a channel dimension (1 for grayscale)

print(f"Training data shape: {x_train.shape}") # Print the shape of the training data
print(f"Testing data shape: {x_test.shape}") # Print the shape of the testing data

# ------------------------------------------------------------
# Step 2: Build the CNN model architecture
# ------------------------------------------------------------
# Typical CNN architecture for MNIST classification

model = models.Sequential([ # Create a Sequential model, where layers are added in order
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Add a 2D convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and specifying input shape
    layers.MaxPooling2D((2, 2)), # Add a 2D max pooling layer with a 2x2 pool size

    layers.Conv2D(64, (3, 3), activation='relu'), # Add another 2D convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
    layers.MaxPooling2D((2, 2)), # Add another 2D max pooling layer with a 2x2 pool size

    layers.Flatten(), # Flatten the output from the convolutional layers into a 1D array
    layers.Dense(128, activation='relu'), # Add a dense (fully connected) layer with 128 units and ReLU activation
    layers.Dense(10, activation='softmax')  # Add the output dense layer with 10 units (for 10 classes) and softmax activation for probabilities
])

# Display model architecture
model.summary() # Print a summary of the model's architecture

# ------------------------------------------------------------
# Step 3: Compile the model
# ------------------------------------------------------------
model.compile(
    optimizer='adam',              # Use the Adam optimizer
    loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy as the loss function (suitable for integer labels)
    metrics=['accuracy']           # Track accuracy during training
)

# ------------------------------------------------------------
# Step 4: Train the model
# ------------------------------------------------------------
# Training with 5 epochs usually achieves >98% accuracy
history = model.fit(
    x_train, y_train,
    epochs=5, # Train for 5 epochs
    validation_data=(x_test, y_test), # Use the test data for validation
    batch_size=64, # Use a batch size of 64
    verbose=2 # Display training progress verbosely (one line per epoch)
)

# ------------------------------------------------------------
# Step 5: Evaluate the model
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) # Evaluate the model on the test data
print(f"\nTest Accuracy: {test_acc * 100:.2f}%") # Print the test accuracy

# ------------------------------------------------------------
# Step 6: Visualize model performance
# ------------------------------------------------------------
plt.figure(figsize=(8, 5)) # Create a figure for the plot
plt.plot(history.history['accuracy'], label='Training Accuracy') # Plot the training accuracy over epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Plot the validation accuracy over epochs
plt.title('Model Accuracy Over Epochs') # Set the title of the plot
plt.xlabel('Epoch') # Set the x-axis label
plt.ylabel('Accuracy') # Set the y-axis label
plt.legend() # Display the legend
plt.show() # Show the plot

# ------------------------------------------------------------
# Step 7: Visualize sample predictions
# ------------------------------------------------------------
# Select 5 random test images
num_samples = 5 # Define the number of samples to visualize
indices = np.random.choice(len(x_test), num_samples, replace=False) # Select random indices from the test set
sample_images = x_test[indices] # Get the sample images
sample_labels = y_test[indices] # Get the sample labels

# Predict using the trained model
predictions = np.argmax(model.predict(sample_images), axis=1) # Make predictions on the sample images and get the predicted class

# Plot the images with predicted and true labels
plt.figure(figsize=(10, 4)) # Create a figure for the plot
for i in range(num_samples): # Loop through the sample images
    plt.subplot(1, num_samples, i + 1) # Create a subplot for each image
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray') # Display the image in grayscale
    plt.title(f"Pred: {predictions[i]}\nTrue: {sample_labels[i]}") # Set the title with predicted and true labels
    plt.axis('off') # Turn off the axes
plt.show() # Show the plot

# ------------------------------------------------------------
# Step 8: Save the trained model
# ------------------------------------------------------------
model.save("mnist_cnn_model.h5") # Save the trained model in HDF5 format
print("\nModel saved as mnist_cnn_model.h5") # Print a confirmation message
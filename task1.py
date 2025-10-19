# ==============================================================
# AI Tools  - Task 1: Classical ML with Scikit-learn
# Dataset: Iris Species Dataset
# Goal: Predict iris species using a Decision Tree Classifier
# ==============================================================

# Import essential libraries for data manipulation, machine learning, and model evaluation
import pandas as pd # Import pandas for data handling (DataFrames)
from sklearn.datasets import load_iris # Import load_iris to get the dataset
from sklearn.model_selection import train_test_split # Import train_test_split for splitting data
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder for encoding target variable
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier for the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report # Import metrics for evaluation

# --------------------------------------------------------------
#  Load and Inspect the Dataset
# --------------------------------------------------------------
# Load the Iris dataset from sklearn's built-in library
iris = load_iris()

# Convert the dataset to a pandas DataFrame for easier manipulation
df = pd.DataFrame(iris.data, columns=iris.feature_names) # Create DataFrame from data with feature names as columns
df['species'] = iris.target  # Add the target column (encoded as 0,1,2 by default in sklearn)

# Display the first few rows of the DataFrame
print("Dataset Preview:") # Print a header for the preview
print(df.head(), "\n") # Print the first 5 rows of the DataFrame

# --------------------------------------------------------------
#  Check for Missing Values
# --------------------------------------------------------------
print("Checking for missing values:") # Print a header for missing value check
print(df.isnull().sum(), "\n") # Print the count of missing values for each column

# No missing values exist in this dataset, but we demonstrate how to handle them if they did:
# df = df.fillna(df.mean()) # Example of filling missing values with the mean of the column

# --------------------------------------------------------------
#  Encode Labels (if needed)
# --------------------------------------------------------------
# The target is already numeric (0,1,2), but for demonstration, we'll show the encoding process
label_encoder = LabelEncoder() # Initialize the LabelEncoder
df['species'] = label_encoder.fit_transform(df['species']) # Fit and transform the 'species' column

# --------------------------------------------------------------
#  Split the Data into Train and Test Sets
# --------------------------------------------------------------
X = df.drop('species', axis=1)  # Define features (X) by dropping the 'species' column
y = df['species']               # Define the target variable (y) as the 'species' column

# Split the data into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Split X and y, 20% for testing, use random_state for reproducibility, stratify to maintain class distribution
)

# Print the number of samples in the training and testing sets
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}\n") # Print the shapes of the resulting sets

# --------------------------------------------------------------
#  Initialize and Train the Decision Tree Model
# --------------------------------------------------------------
# Create a Decision Tree Classifier instance
model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=3) # Initialize the model with parameters

# Train (fit) the model using the training data
model.fit(X_train, y_train) # Train the model on the training features and target

# --------------------------------------------------------------
#  Make Predictions
# --------------------------------------------------------------
# Make predictions on the test set
y_pred = model.predict(X_test) # Predict the target variable for the test features

# --------------------------------------------------------------
#  Evaluate Model Performance
# --------------------------------------------------------------
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy score
precision = precision_score(y_test, y_pred, average='macro') # Calculate the precision score with macro averaging
recall = recall_score(y_test, y_pred, average='macro') # Calculate the recall score with macro averaging

# Print the evaluation metrics
print("Model Evaluation Metrics:") # Print a header for evaluation metrics
print(f"Accuracy : {accuracy:.3f}") # Print the calculated accuracy
print(f"Precision: {precision:.3f}") # Print the calculated precision
print(f"Recall   : {recall:.3f}\n") # Print the calculated recall

# Print the detailed classification report for further insight
print("Detailed Classification Report:") # Print a header for the classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names)) # Print the classification report with target names

# --------------------------------------------------------------
#  Visualize Decision Tree Structure
# --------------------------------------------------------------
# Import libraries for visualization
from sklearn import tree # Import the tree module
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Create a figure and axes for the plot
plt.figure(figsize=(10,6)) # Set the size of the figure

# Plot the decision tree
tree.plot_tree(model, feature_names=iris.feature_names, # Plot the tree, using feature names
               class_names=iris.target_names, filled=True, rounded=True) # Use class names, fill nodes with color, round node corners
plt.title("Decision Tree Classifier - Iris Dataset") # Set the title of the plot
plt.show() # Display the plot
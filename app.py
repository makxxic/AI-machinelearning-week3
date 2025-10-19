import streamlit as st # Import the streamlit library for building web applications
import numpy as np # Import numpy for numerical operations
import tensorflow as tf # Import tensorflow for building and loading the deep learning model
from PIL import Image, ImageOps # Import Image and ImageOps from Pillow for image manipulation
from streamlit_drawable_canvas import st_canvas # Import st_canvas for the drawing functionality in streamlit
from tensorflow.keras.datasets import mnist # Import the MNIST dataset from tensorflow.keras
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs
# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier", # Set the title of the web page
    page_icon="🧮", # Set the favicon of the web page
    layout="wide" # Set the layout of the page to wide
)

st.title("MNIST Handwritten Digit Classifier") # Display the main title of the app
st.markdown("### Draw a digit (0–9), predict it, and explore how the CNN works!") # Display a markdown header and text

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    model = tf.keras.models.load_model("task2TF.py") # Load the pre-trained MNIST CNN model
    return model # Return the loaded model

model = load_model() # Load the model

# -----------------------------------------
# DRAWING CANVAS
# -----------------------------------------
st.subheader("Draw a Digit") # Display a subheader for the drawing section
st.write("Use your mouse (or touch) to draw a number below:") # Provide instructions for the user

canvas_result = st_canvas(
    fill_color="white", # Set the background color of the canvas
    stroke_width=20, # Set the width of the drawing stroke
    stroke_color="black", # Set the color of the drawing stroke
    background_color="white", # Set the background color of the canvas
    width=280, # Set the width of the canvas in pixels
    height=280, # Set the height of the canvas in pixels
    drawing_mode="freedraw", # Set the drawing mode to freehand drawing
    key="canvas" # Assign a unique key to the canvas element
)

col1, col2 = st.columns([1, 1]) # Create two columns for layout

with col1: # Work within the first column
    if st.button("Predict Digit"): # Create a button to trigger prediction
        if canvas_result.image_data is not None: # Check if there is any drawing on the canvas
            # Preprocess image
            img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8)) # Convert the canvas image data to a PIL Image and invert colors
            img = img.resize((28, 28)) # Resize the image to 28x28 pixels
            img = ImageOps.grayscale(img) # Convert the image to grayscale
            img_array = np.array(img) / 255.0 # Convert the image to a numpy array and normalize pixel values
            img_array = img_array.reshape(1, 28, 28, 1) # Reshape the array to match the model's input shape

            # Predict
            prediction = model.predict(img_array) # Get the model's prediction for the drawn digit
            pred_class = np.argmax(prediction) # Get the predicted digit class (the one with the highest probability)
            confidence = np.max(prediction) * 100 # Calculate the confidence of the prediction as a percentage

            st.success(f"Predicted Digit: {pred_class}") # Display the predicted digit
            st.info(f"Confidence: {confidence:.2f}%") # Display the confidence of the prediction

            st.bar_chart(prediction[0]) # Display a bar chart showing the probability distribution for each digit class
        else:
            st.warning("Please draw a digit before predicting.") # Display a warning if no digit is drawn

with col2: # Work within the second column
    if st.button("Clear Canvas"): # Create a button to clear the canvas
        st.experimental_rerun() # Rerun the app to clear the canvas

# -----------------------------------------
# SAMPLE PREDICTION GALLERY
# -----------------------------------------
st.subheader("Sample Prediction Gallery") # Display a subheader for the sample predictions section

@st.cache_data # Cache the data loading to avoid reloading on every rerun
def get_mnist_samples(n=5):
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the MNIST dataset
    return x_test[:n], y_test[:n] # Return the first n images and labels from the test set

x_samples, y_samples = get_mnist_samples() # Get the sample images and labels
sample_cols = st.columns(5) # Create five columns for displaying sample images
for i, col in enumerate(sample_cols): # Iterate through the sample images and columns
    img = x_samples[i] # Get the current sample image
    true_label = y_samples[i] # Get the true label for the current image
    img_input = img.reshape(1, 28, 28, 1) / 255.0 # Reshape and normalize the image for model input
    pred = np.argmax(model.predict(img_input)) # Get the model's prediction for the sample image

    with col: # Work within the current column
        st.image(img, width=70, caption=f"True: {true_label} | Pred: {pred}") # Display the sample image with its true and predicted labels

# -----------------------------------------
# MODEL ARCHITECTURE INSIGHT
# -----------------------------------------
st.subheader("CNN Model Architecture Overview") # Display a subheader for the model architecture section

with st.expander("View Model Summary"): # Create an expandable section for the model summary
    # Capture model summary as text
    from io import StringIO # Import StringIO to capture text output
    stream = StringIO() # Create a StringIO object
    model.summary(print_fn=lambda x: stream.write(x + "\n")) # Print the model summary to the StringIO object
    summary_str = stream.getvalue() # Get the string value from the StringIO object
    st.text(summary_str) # Display the model summary as text

st.caption("This CNN model is trained on the MNIST dataset to classify handwritten digits (0–9).") # Display a caption

# -----------------------------------------
# VISUALIZE FEATURE MAPS (LAYER ACTIVATIONS)
# -----------------------------------------
st.subheader("Visualizing CNN Layer Activations") # Display a subheader for feature map visualization

st.markdown("See what the model 'sees' at different layers when classifying a digit:") # Provide an explanation for feature map visualization

# Load one sample image from MNIST test set
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Load the MNIST dataset
sample_img = x_test[0].reshape(1, 28, 28, 1) / 255.0 # Get the first test image, reshape and normalize it
st.image(x_test[0], width=100, caption=f"Input Image (True Label: {y_test[0]})") # Display the input image with its true label

# Choose layer
layer_names = [layer.name for layer in model.layers if "conv" in layer.name or "max_pool" in layer.name] # Get the names of convolutional and max pooling layers
layer_choice = st.selectbox("Select layer to visualize:", layer_names) # Create a selectbox to choose a layer for visualization

if layer_choice: # Check if a layer is selected
    layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_choice).output) # Create a new model that outputs the activations of the selected layer
    activations = layer_model.predict(sample_img) # Get the activations of the selected layer for the sample image

    n_features = activations.shape[-1] # Get the number of features (filters) in the selected layer
    n_cols = 8 # Define the number of columns for displaying feature maps
    n_rows = int(np.ceil(n_features / n_cols)) # Calculate the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10)) # Create a figure and axes for plotting
    for i in range(n_features): # Iterate through each feature map
        row, col = divmod(i, n_cols) # Calculate the row and column for the current feature map
        ax = axes[row, col] # Get the current axis
        ax.imshow(activations[0, :, :, i], cmap="gray") # Display the feature map in grayscale
        ax.axis("off") # Turn off the axes
    st.pyplot(fig) # Display the plot in Streamlit

st.caption("Feature maps reveal how convolutional layers detect edges, curves, and shapes of digits.") # Display a caption

# -----------------------------------------
# SIDEBAR INFO
# -----------------------------------------
st.sidebar.header("About This App") # Display a header in the sidebar
st.sidebar.markdown("""
This interactive app demonstrates a **Convolutional Neural Network (CNN)**
trained on the **MNIST handwritten digits dataset**.

**Features:**
- Draw and predict digits
- Real-time confidence scores
- CNN architecture overview
- Feature map visualization
- Sample test predictions

**Frameworks Used:**
- TensorFlow / Keras
- Streamlit
- Streamlit Drawable Canvas
""") # Display markdown text in the sidebar

st.sidebar.caption("Built by an AI Engineer | Powered by TensorFlow") # Display a caption in the sidebar
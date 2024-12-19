import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('beans_classification_model.keras')

# Define class names based on the beans dataset
class_names = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']

# Title and instructions
st.title('Bean Leaf Disease Classification')
st.write("""
Upload an image of a bean leaf, and the model will predict whether the leaf is healthy or has a disease (Angular Leaf Spot or Bean Rust).
""")

# File uploader for bean leaf image
uploaded_file = st.file_uploader("Choose a bean leaf image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Bean Leaf Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img = img.resize((224, 224))  # Resize the image to 224x224 as required by MobileNet
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]

    # Display prediction result
    st.write(f"Prediction: **{predicted_label}**")

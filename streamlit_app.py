import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('tumor_detection_model.h5')

st.title("🩺 Breast Cancer AI Detection App")

st.write(
    "Welcome! This application uses an AI model to predict the presence of breast cancer in images. "
    "You can upload a set of mammogram images and see the predictions of our AI model. "
    "This app is intended for educational purposes and should not be used for medical diagnosis."
)

# File uploader for images
uploaded_files = st.file_uploader("Upload mammogram images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write("## Uploaded Images")
    
    data = []  # List to store prediction results
    
    for uploaded_file in uploaded_files:
        # Load the image
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Get prediction from the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        probability = np.max(prediction)
        
        # Determine the predicted label
        label = "Tumor Detected" if predicted_class == 1 else "No Tumor"
        
        # Append to data list
        data.append({
            "Image Name": uploaded_file.name,
            "Prediction": label,
            "Probability": f"{probability:.2f}"
        })
        
        # Display the image and prediction
        st.image(uploaded_file, caption=f"Prediction: {label} (Probability: {probability:.2f})", use_column_width=True)

    # Display results in a table
    st.write("## Summary of Predictions")
    df = pd.DataFrame(data)
    st.write(df)

    # Visualize annotation progress
    st.divider()
    st.write("### Annotation Summary")
    tumor_count = len(df[df["Prediction"] == "Tumor Detected"])
    total_count = len(df)
    tumor_percentage = f"{(tumor_count / total_count) * 100:.2f}%"
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Number of Images with Tumors", tumor_count)
    with col2:
        st.metric("Tumor Detection Rate", tumor_percentage)

    st.bar_chart(df["Prediction"].value_counts().reset_index(), x="index", y="Prediction")

st.write(
    "Thank you for using the Breast Cancer AI Detection App. Remember, this tool is for educational purposes and "
    "is not intended to replace professional medical diagnosis. If you have any concerns, please consult a medical professional."
)

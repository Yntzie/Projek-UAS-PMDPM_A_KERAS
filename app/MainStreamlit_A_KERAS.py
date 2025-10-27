import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()
class_names = ['Kerak Telor', 'Papeda', 'Bika Ambon', 'Plecing Kangkung']

# Function to preprocess and classify image
def classify_image(image_path):
    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim= tf.expand_dims (input_image_array, 0)

        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax (predictions[0]) # Apply softmax for probability

        # Get class with highest confidence
        class_idx= np.argmax(result)
        confidence_scores = result.numpy()
        return class_names [class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)
    
# Function to create a custom progress bar
def custom_progress_bar (confidence, color1, color2, color3, color4):
    percentage1 = confidence[0] * 100 # Kerak Telor
    percentage2 = confidence[1] * 100 # Papeda
    percentage3 = confidence[2] * 100 # Bika Ambon
    percentage4 = confidence[3] * 100 # Plecing Kangkung

    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;"> 
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;"> 
            {percentage2:.2f}%
        </div>
        <div style="width: {percentage3:.2f}%; background: {color3}; color: white; text-align: center; height: 24px; float: left;"> 
            {percentage3:.2f}%
        </div>
        <div style="width: {percentage4:.2f}%; background: {color4}; color: white; text-align: center; height: 24px; float: left;"> 
            {percentage4:.2f}%
        </div>
    </div>
    """
    st.sidebar.markdown (progress_html, unsafe_allow_html=True)

# StreamLit UI
st.title("Prediksi Makanan Tradisional Indonesia")

# Upload multiple files in the main page
uploaded_files = st.file_uploader ("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Sidebar for prediction button and results
if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f: 
                f.write(uploaded_file.getbuffer())

            # Perform prediction
            label, confidence = classify_image (uploaded_file.name)

            if label != "Error":
                # Define colors for the bar and Label
                first_color="#FF4136" # Red for "Kerak Telor" 
                second_color="#fff236" # Yellow for "Papeda"
                third_color="#58ff36" # Green for "Bika Ambon"
                fourth_color="#36a2ff" # Blue for "Plecing Kangkung"
                if(label == "Kerak Telor"):
                    label_color = first_color
                elif(label == "Papeda"):
                    label_color = second_color
                elif(label == "Bika Ambon"):
                    label_color = third_color
                else:
                    label_color = fourth_color

                # Display prediction results
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.markdown (f" <h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)

                # Display confidence scores
                st.sidebar.write("**Confidence:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")

                # Display custom progress bar
                custom_progress_bar (confidence, first_color, second_color, third_color, fourth_color)
                st.sidebar.write("---")
            
            else:
                st.sidebar.error(f" Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

# Preview images in the main page
if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)

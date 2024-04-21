import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model

def custom_adam(*args, **kwargs):
    return tf.keras.optimizers.Adam(*args, **kwargs)

color_model = load_model('color_classifier_model.h5', custom_objects={'custom_adam': custom_adam()})
clothing_model = load_model('clothing_classifier_model.h5', custom_objects={'custom_adam': custom_adam()})



# Load label encoders
color_label_encoder = LabelEncoder()
color_label_encoder.classes_ = np.load('colors.npz')['arr_0']

clothing_label_encoder = LabelEncoder()
clothing_label_encoder.classes_ = np.load('classes.npy')

st.title('Result')

# Upload image
uploaded_file = st.file_uploader("Upload an image to classify the clothing and its color", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file from the uploader
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Resize and preprocess the image
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    
    # Predict color
    color_pred = color_model.predict(np.array([img]))
    color_label = color_label_encoder.inverse_transform([np.argmax(color_pred)])[0]
    
    # Predict clothing
    clothing_pred = clothing_model.predict(np.array([img]))
    clothing_label = clothing_label_encoder.inverse_transform([np.argmax(clothing_pred)])[0]

    # Store results in session state
    st.session_state.color = color_label
    st.session_state.clothing = clothing_label
    st.session_state.uploaded_image = image

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display classification results
    st.write(f"Color: {color_label}")
    st.write(f"Clothing: {clothing_label}")

    # Button to upload another image
    if st.button('Upload another image'):
        st.session_state.color = ''
        st.session_state.clothing = ''
        st.session_state.uploaded_image = None

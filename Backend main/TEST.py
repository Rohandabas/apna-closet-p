import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Title of the app and navigation bar
st.title('APNA CLOSET')

# Navigation bar
nav_selection = st.sidebar.radio("NAVIGATION", ["Home", "About", "Contact"])

if nav_selection == "Home":
    st.write("Welcome to the Outfit Recommendation System")

elif nav_selection == "About":
    st.header("About")
    st.write(
        """
        This app recommends similar fashion items based on the features extracted from the uploaded image.
        It utilizes a pre-trained ResNet50 model to extract image features and a nearest neighbors algorithm 
        to find visually similar items from the dataset.
        """
    )
    st.subheader("How it works")
    st.write(
        """
        1. *Upload Image*: Choose an image of a fashion item you like.
        2. *Get Recommendations*: The system will recommend visually similar fashion items based on the features of the uploaded image.
        3. **This will also gives you the rgb color code for the recommended images.
        """
    )

elif nav_selection == "Contact":
    st.header("Contact Us")
    st.write(
        """
        If you have any questions or feedback, feel free to reach out to us:

        - *Email*: arnimsaxena4nis@gmail.com
        - *Phone*: +91 (6398640349)
        - *Address*: ORS, Greater Noida, India
        """
    )


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file with animation
        display_image = Image.open(uploaded_file)
        st.image(display_image, use_container_width=True, caption="Uploaded Image")

        # Extract features from uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations based on extracted features
        indices = recommend(features, feature_list)

        # Display recommended images with animation
        st.subheader("Recommended Images")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]], use_container_width=True)
        with col2:
            st.image(filenames[indices[0][1]], use_container_width=True)
        with col3:
            st.image(filenames[indices[0][2]], use_container_width=True)
        with col4:
            st.image(filenames[indices[0][3]], use_container_width=True)
        with col5:
            st.image(filenames[indices[0][4]], use_container_width=True)
    else:
        st.header("An error occurred during file upload")

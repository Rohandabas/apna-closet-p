import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

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
nav_selection = st.sidebar.radio(" N A V I G A T I O N", ["Home", "About", "Contact"])

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

        - *Email*: arnimsaxena4@gmail.com
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
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size here
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


# Color extraction function
def extract_colors(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to reduce computation
    img = cv2.resize(img, (100, 100))

    # Flatten image to 1D array
    pixels = img.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

    # Perform K-Means clustering
    _, _, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = np.uint8(centers)

    return centers.tolist()


# Convert color codes to hexadecimal
def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])


# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        st.subheader("Uploaded Image")
        display_image = Image.open(uploaded_file)
        st.image(display_image, width=250, caption="Uploaded Image")  # Adjust width here

        # Extract features from uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations based on extracted features
        indices = recommend(features, feature_list)

        # Display recommended images
        st.subheader("Recommended Images")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            recommended_image = Image.open(filenames[indices[0][0]])
            st.image(recommended_image, width=90, caption="Image 1")  # Adjust width here
            colors = extract_colors(filenames[indices[0][0]])
            st.write("Dominant Colors (RGB):")
            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                st.color_picker(f"color_{indices[0][0]}{i}", hex_color, key=f"color{indices[0][0]}_{i}")

        with col2:
            recommended_image = Image.open(filenames[indices[0][1]])
            st.image(recommended_image, width=90, caption="Image 2")  # Adjust width here
            colors = extract_colors(filenames[indices[0][1]])
            st.write("Dominant Colors (RGB):")
            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                st.color_picker(f"color_{indices[0][1]}{i}", hex_color, key=f"color{indices[0][1]}_{i}")

        with col3:
            recommended_image = Image.open(filenames[indices[0][2]])
            st.image(recommended_image, width=90, caption="Image 3")  # Adjust width here
            colors = extract_colors(filenames[indices[0][2]])
            st.write("Dominant Colors (RGB):")
            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                st.color_picker(f"color_{indices[0][2]}{i}", hex_color, key=f"color{indices[0][2]}_{i}")

        with col4:
            recommended_image = Image.open(filenames[indices[0][3]])
            st.image(recommended_image, width=90, caption="Image 4")  # Adjust width here
            colors = extract_colors(filenames[indices[0][3]])
            st.write("Dominant Colors (RGB):")
            for i, color in enumerate(colors):
                hex_color = rgb_to_hex(color)
                st.color_picker(f"color_{indices[0][3]}{i}", hex_color, key=f"color{indices[0][3]}_{i}")



        with col5:
            recommended_image = Image.open(filenames[indices[0][4]])
            st.image(recommended_image, width=90, caption="Image 5")  # Adjust width here
            colors = extract_colors(filenames[indices[0][4]])
            st.write("Dominant Colors (RGB):")
            for color in colors:
                hex_color = rgb_to_hex(color)
                st.color_picker("", hex_color)
    else:
        st.header("An error occurred during file upload")

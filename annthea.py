import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

tf.get_logger().setLevel('ERROR')

# Load the model
model = load_model("keras_model.h5")

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Load the dataset for recommendation
dataset = pd.read_csv("obat_pengobatan.csv")

# Load the dataset
train_data = pd.read_csv('obat_pengobatan.csv')

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply TF-IDF vectorizer to the "Obat/Pengobatan" column
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Obat/Pengobatan'])

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(acne_level, top_n=1):
    # Get the index of the acne level in the dataset
    level_indices = train_data[train_data['Tingkat'] == acne_class].index
    
    if len(level_indices) == 0:
        return []
    
    level_index = level_indices[0]
    
    # Get the cosine similarity scores for the level
    similarity_scores = list(enumerate(cosine_sim_matrix[level_index]))
    
    # Sort the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    top_recommendations = similarity_scores[1:top_n+1]
    
    return top_recommendations

# Streamlit app
st.title("Deteksi dan Rekomendasi Pengobatan Jerawat")

# Upload image
uploaded_image = st.file_uploader("Unggah gambar jerawat")
if uploaded_image is not None:
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Preprocess the image
    image = Image.open(uploaded_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Make acne prediction
    acne_prediction = model.predict(data)
    acne_index = np.argmax(acne_prediction)
    acne_class = class_names[acne_index].strip()
    acne_confidence = acne_prediction[0][acne_index]
    
    # Display acne prediction and confidence score
    st.image(image, caption='Nama Gambar', use_column_width=True)
    st.subheader("Prediksi Jerawat")
    st.write("Kelas Jerawat:", acne_class)
    st.write("Skor Kepercayaan Jerawat:", acne_confidence)
    
    # Tambahkan input untuk acne_class
    acne_class = st.text_input("Masukkan tingkat jerawat (Misal: lv0, lv1, lv2)")

    # Gunakan acne_class untuk mendapatkan rekomendasi
    recommendations = get_recommendations(acne_class)
    for recommendation in recommendations:
        st.write("Obat/Pengobatan:", train_data.iloc[recommendation[0]]['Obat/Pengobatan'])
        st.write("Penggunaan yang Disarankan:", train_data.iloc[recommendation[0]]['Penggunaan yang Disarankan'])
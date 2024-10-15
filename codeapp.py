import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.ndimage import center_of_mass
from io import BytesIO
from PIL import Image

def initialize_centroids(pixels, n_clusters):
    np.random.seed(42)
    random_idxs = np.random.choice(len(pixels), n_clusters, replace=False)
    return pixels[random_idxs]

def assign_clusters(pixels, centroids):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(pixels, labels, n_clusters):
    new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])
    return new_centroids

def manual_kmeans(pixels, n_clusters, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(pixels, n_clusters)
    for i in range(max_iters):
        labels = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, n_clusters)
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        centroids = new_centroids
    return labels, centroids

def preprocess_image(image, size=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

def cluster_image(image, n_clusters):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    labels, centroids = manual_kmeans(pixel_values, n_clusters)
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image, centroids, pixel_values, labels

def visualize_segmentation(image, segmented_image, n_clusters):
    labeled_image = np.zeros_like(image)
    colors = np.random.randint(0, 255, (n_clusters, 3))
    for cluster_num in range(n_clusters):
        labeled_image[segmented_image == cluster_num] = colors[cluster_num]
    return labeled_image

def evaluate_clustering(pixel_values, labels):
    return silhouette_score(pixel_values, labels)

def save_image(image):
    im_pil = Image.fromarray(image)
    buf = BytesIO()
    im_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

st.set_page_config(page_title='Image Clustering App', layout='wide')

st.title('Image Clustering with K-Means')
st.write("This app allows you to perform image segmentation using K-Means clustering and visualize the results for different numbers of clusters.")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.write("Adjust clustering parameters:")
    max_clusters = st.slider("Select maximum number of clusters:", 2, 10, 5)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    preprocessed_image = preprocess_image(image)

    st.image(preprocessed_image, caption='Uploaded Image', use_column_width=True)

    best_silhouette = -1
    best_n_clusters = None
    best_segmented_image = None

    st.write("### Clustering Results:")
    progress_bar = st.progress(0)

    for i, n_clusters in enumerate(range(2, max_clusters + 1)):
        segmented_img, centroids, pixel_values, labels = cluster_image(preprocessed_image, n_clusters)
        silhouette = evaluate_clustering(pixel_values, labels)

        labeled_img = visualize_segmentation(preprocessed_image, segmented_img, n_clusters)
        st.image(labeled_img, caption=f"{n_clusters} Clusters - Silhouette Score: {silhouette:.4f}", use_column_width=True)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_n_clusters = n_clusters
            best_segmented_image = labeled_img

        progress_bar.progress((i + 1) / (max_clusters - 1))

    st.write(f"### Best Result: {best_n_clusters} clusters with a silhouette score of {best_silhouette:.4f}.")
    st.image(best_segmented_image, caption=f"Best Cluster: {best_n_clusters} clusters", use_column_width=True)

    st.sidebar.header("Download Best Segmented Image")
    btn = st.sidebar.download_button(
        label="Download Image",
        data=save_image(best_segmented_image),
        file_name="segmented_image.png",
        mime="image/png"
    )

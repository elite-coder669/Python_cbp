import numpy as np
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split

def load_data_from_files():
    """Prompts the user to select .npy files for images and labels, loads, and preprocesses them."""
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to select dataset files (.npy format)
    files_path = filedialog.askopenfilenames(
        title="Select your dataset and label files",
        filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")]
    )
    
    # Ensure two files are selected (images and labels)
    if files_path and len(files_path) == 2:
        print(f"Successfully loaded dataset from {files_path}")
        image_path, labels_path = files_path
        images = np.load(image_path)  # Load images
        labels = np.load(labels_path)  # Load labels

        # Preprocess the images (flatten and normalize)
        flattened_images = flatten_images(images)
        normalized_images = normalize_data(flattened_images)

        return normalized_images, labels
    else:
        print("Failed loading dataset. Please select exactly two .npy files (images and labels).")
        return None, None

def flatten_images(images):
    """Flattens 2D images into 1D arrays."""
    return images.reshape(images.shape[0], -1)

def normalize_data(data):
    """Normalizes the data using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def stratified_train_test_split(data, labels, test_size=0.2):
    """
    Splits the data into training and testing sets using stratification.
    """
    return train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=42)

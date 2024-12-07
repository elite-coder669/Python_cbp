import numpy as np
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog

def load_data_from_files():
    """Prompts the user to select .npy files for images and labels, and loads them."""
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
        return images, labels
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

def save_data_to_location(images, labels, save_dir):
    """Saves processed images and labels to the specified directory."""
    if images is not None and labels is not None:
        # Save the processed images and labels as .npy files
        np.save(f"{save_dir}/pca_processed_images.npy", images)
        np.save(f"{save_dir}/pca_processed_labels.npy", labels)
        print(f"Data saved to {save_dir}")
    else:
        print("No data to save.")

# Load images and labels from selected .npy files
images, labels = load_data_from_files()

if images is not None and labels is not None:
    # Flatten and normalize the images
    flattened_images = flatten_images(images)
    normalized_images = normalize_data(flattened_images)

    # Prompt for directory to save the processed data
    save_dir = filedialog.askdirectory(title="Select Directory to Save Processed Data")
    
    if save_dir:
        # Save the processed data to the selected directory
        save_data_to_location(normalized_images, labels, save_dir)
else:
    print("Dataset loading was unsuccessful. Please try again.")

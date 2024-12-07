import numpy as np
from data_preprocessing import load_data_from_files

def display_data_as_matrices():
    # Load the data
    normalized_images, labels = load_data_from_files()

    if normalized_images is not None and labels is not None:
        # Configure NumPy to print full matrices without truncation
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        # Display the image data matrix
        # print("Normalized Images Matrix:")
        # print(normalized_images)

        # # Display the labels matrix
        # print("\nLabels Matrix:")
        # print(labels)
        data = np.column_stack((normalized_images,labels))
        print(data)
        # Reset NumPy print options to default (optional)
        # np.set_printoptions(threshold=1000, linewidth=75)
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    display_data_as_matrices()
## prints the complete numpy arrays.
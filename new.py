import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tk, filedialog

def select_files():
    """
    Opens a file dialog to select image and label .npy files.
    Returns the paths of the selected image and label files.
    """
    root = Tk()
    root.withdraw()  # Hide the root Tkinter window
    file_paths = filedialog.askopenfilenames(
        title="Select Image and Label Files",
        filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")]
    )
    if len(file_paths) != 2:
        print("Error: Please select exactly two .npy files (images and labels).")
        return None, None
    return file_paths

class ImageNavigator:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.index = 0  # Start with the first image

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.image_plot = self.ax.imshow(self.images[self.index], cmap='gray')
        self.title = self.ax.set_title(f"Image {self.index + 1}/{len(self.images)} - Label: {self.labels[self.index]}")
        self.ax.axis('off')

        # Connect arrow key events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()

    def update_image(self):
        """Updates the displayed image and title."""
        self.image_plot.set_data(self.images[self.index])
        self.title.set_text(f"Image {self.index + 1}/{len(self.images)} - Label: {self.labels[self.index]}")
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Handles key press events to navigate images."""
        if event.key == 'right':  # Right arrow key
            self.index = (self.index + 1) % len(self.images)  # Wrap around
        elif event.key == 'left':  # Left arrow key
            self.index = (self.index - 1) % len(self.images)  # Wrap around
        self.update_image()

if __name__ == "__main__":
    # Prompt the user to select image and label files
    image_file, label_file = select_files()
    
    if image_file and label_file:
        # Load the images and labels
        images = np.load(image_file)
        labels = np.load(label_file)
        
        # Check if the number of images matches the number of labels
        if len(images) != len(labels):
            print("Error: The number of images and labels do not match!")
        else:
            # Start the image navigator
            navigator = ImageNavigator(images, labels)

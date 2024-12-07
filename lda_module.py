from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

def apply_lda(data, labels, n_components=None):
    """
    Applies Linear Discriminant Analysis (LDA) to the data.

    Parameters:
    - data: The input data (features).
    - labels: The class labels for supervised LDA.
    - n_components: Number of components to keep. If None, defaults to min(n_features, n_classes - 1).

    Returns:
    - lda: Trained LDA model.
    - transformed_data: Transformed data with LDA.
    """
    n_features = data.shape[1]
    n_classes = len(set(labels))
    max_components = min(n_features, n_classes - 1)

    if n_components is None or n_components > max_components:
        n_components = max_components
        print(f"Adjusted n_components to {n_components}, as it cannot exceed min(n_features, n_classes - 1).")

    lda = LDA(n_components=n_components)
    transformed_data = lda.fit_transform(data, labels)
    return lda, transformed_data


def visualize_lda_components(lda, h, w, n_row=2, n_col=3):
    """
    Visualizes the LDA components (similar to eigenfaces in PCA).

    Parameters:
    - lda: Trained LDA model.
    - h, w: Height and width of the original images.
    - n_row: Number of rows in the grid.
    - n_col: Number of columns in the grid.
    """
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        if i >= lda.scalings_.shape[1]:
            break
        lda_component = lda.scalings_[:, i].reshape(h, w)
        ax.imshow(lda_component, cmap='viridis')  # Use a colorful map to distinguish from PCA
        ax.axis('off')
        ax.set_title(f"LDA Component {i+1}")
    plt.tight_layout()
    plt.show()

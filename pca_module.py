from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def apply_pca(data, n_components=100):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return pca, transformed_data

def visualize_eigenfaces(pca, h, w, n_row=3, n_col=5):
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        eigenface = pca.components_[i].reshape(h, w)
        ax.imshow(eigenface, cmap='gray')
        ax.axis('off')
    plt.show()

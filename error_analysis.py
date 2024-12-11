import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def analyze_errors(y_test, y_pred, X_test, h, w):
    """Analyze and visualize the classification errors."""
    errors = np.where(y_test != y_pred)[0]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(errors):
            idx = errors[i]
            # ax.imshow(X_test[idx].reshape(h, w), cmap='gray')
            ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
            ax.axis('off')
    plt.show()

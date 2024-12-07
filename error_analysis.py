import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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

def analyze_classwise_performance(y_test, y_pred):
    """Print class-wise performance metrics."""
    print("Class-wise performance metrics:")
    print(classification_report(y_test, y_pred))

def plot_classwise_metrics(y_test, y_pred):
    """Plot class-wise recall metrics."""
    # Generate classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Dynamically get all class labels from y_test and y_pred
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    
    # Prepare recall values for plotting
    recall = []
    for label in unique_labels:
        str_label = str(int(label))  # Labels in classification report are strings
        if str_label in report:
            recall.append(report[str_label]['recall'])
        else:
            recall.append(0)  # Assign 0 recall for missing classes
    
    # Create class labels for visualization
    class_labels = [f"Class {int(label)}" for label in unique_labels]
    
    # Debug information for verification
    print("Unique Labels:", unique_labels)
    print("Recall Values:", recall)
    
    # Plot class-wise recall
    plt.figure(figsize=(16, 8))
    plt.bar(class_labels, recall, color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Recall")
    plt.title("Class-wise Recall Metrics")
    plt.tight_layout()
    plt.show()

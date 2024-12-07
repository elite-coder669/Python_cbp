from sklearn.model_selection import train_test_split
from data_preprocessing import load_data_from_files
from lda_module import apply_lda, visualize_lda_components
from classification_module import train_classifier, predict_faces
from evaluation_module import evaluate_model, plot_confusion_matrix
from error_analysis import analyze_errors
from utilities import save_model
import numpy as np

def execute_pipeline_with_lda():
    try:
        # Load and preprocess data
        normalized_images, labels = load_data_from_files()
        print(f"Data shape: {normalized_images.shape}, Labels shape: {labels.shape}")

        h, w = int(np.sqrt(normalized_images.shape[1])), int(np.sqrt(normalized_images.shape[1]))

        # Apply LDA
        lda, X_lda = apply_lda(normalized_images, labels, n_components=50)
        print(f"LDA transformed data shape: {X_lda.shape}")
        visualize_lda_components(lda, h, w)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_lda, labels, test_size=0.2, random_state=42)
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        # Train classifier
        classifier = train_classifier(X_train, y_train, model_type='svm')
        print("Classifier trained successfully.")

        # Save the trained model
        save_model(classifier, path='trained_face_recognition_model_with_lda.joblib')
        print("Model saved successfully.")

        # Predict and evaluate
        y_pred = predict_faces(classifier, X_test)
        evaluate_model(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)
        analyze_errors(y_test, y_pred, X_test, normalized_images, h, w)

        print("Pipeline executed successfully.")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    execute_pipeline_with_lda()


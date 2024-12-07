import os
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data_from_files
from pca_module import apply_pca, visualize_eigenfaces
from classification_module import train_classifier, predict_faces
from evaluation_module import evaluate_model, plot_confusion_matrix
from error_analysis import analyze_errors
from utilities import save_model, load_model
import numpy as np

def execute_pipeline():
    # Load and preprocess data
    normalized_images, labels = load_data_from_files()
    
    if normalized_images is not None and labels is not None:
        h, w = int(np.sqrt(normalized_images.shape[1])), int(np.sqrt(normalized_images.shape[1]))  # Image height and width

        # Apply PCA
        pca, X_pca = apply_pca(normalized_images, n_components=100)
        visualize_eigenfaces(pca, h, w)

        model_path = 'trained_face_recognition_model.joblib'
        
        # Check if model exists
        if os.path.exists(model_path):
            user_input = input(f"Model '{model_path}' already exists. Do you want to replace it? (yes/no): ").strip().lower()
            if user_input == 'yes':
                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2, random_state=42)

                # Train classifier
                classifier = train_classifier(X_train, y_train, model_type='svm')

                # Save the trained model
                save_model(classifier, path=model_path)

                print("Model replaced and saved as 'trained_face_recognition_model.joblib'.")
            else:
                # Load the existing model
                classifier = load_model(model_path)
                print(f"Using existing model '{model_path}' for demonstration.")

        else:
            # Model does not exist, so train and save a new one
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_pca, labels, test_size=0.2, random_state=42)

            # Train classifier
            classifier = train_classifier(X_train, y_train, model_type='svm')

            # Save the trained model
            save_model(classifier, path=model_path)

            print("Model trained and saved as 'trained_face_recognition_model.joblib'.")

        # Predict and evaluate
        y_pred = predict_faces(classifier, X_test)
        evaluate_model(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)
        analyze_errors(y_test, y_pred, X_test, normalized_images, h, w)

        print("Pipeline executed and model saved as 'trained_face_recognition_model.joblib'")

    else:
        print("Data loading and preprocessing were unsuccessful. Please try again.")

if __name__ == "__main__":
    execute_pipeline()

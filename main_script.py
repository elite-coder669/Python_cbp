from sklearn.model_selection import train_test_split
from data_preprocessing import load_data_from_files, stratified_train_test_split
from pca_module import apply_pca, visualize_eigenfaces
from classification_module import train_classifier, predict_faces
from evaluation_module import evaluate_model, plot_confusion_matrix
from error_analysis import analyze_errors, analyze_performance, plot_performance_metrics
from utilities import save_model
import numpy as np

def execute_pipeline():
    # Load and preprocess data
    normalized_images, labels = load_data_from_files()
    
    if normalized_images is not None and labels is not None:
        # Combine data and labels into a single dataset
        data = np.column_stack((normalized_images, labels))
        
        # Shuffle combined data to ensure randomness
        np.random.shuffle(data)
        
        # Separate features and labels after shuffling
        normalized_images, labels = data[:, :-1], data[:, -1]

        # Calculate image height and width
        h, w = int(np.sqrt(normalized_images.shape[1])), int(np.sqrt(normalized_images.shape[1]))

        # Apply PCA for dimensionality reduction
        pca, X_pca = apply_pca(normalized_images, n_components=100)
        visualize_eigenfaces(pca, h, w)

        # Split the dataset using stratified sampling
        X_train, X_test, y_train, y_test = stratified_train_test_split(X_pca, labels)

        # Train classifier with class weights
        classifier = train_classifier(X_train, y_train, model_type='svm')

        # Save the trained model
        # save_model(classifier, path='trained_face_recognition_model.joblib')

        # Predict and evaluate
        y_pred = predict_faces(classifier, X_test)
        evaluate_model(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)
        analyze_errors(y_test, y_pred, X_test, h, w)

        # Overall performance analysis (accuracy, precision, recall, F1 score)
        analyze_performance(y_test, y_pred)
        plot_performance_metrics(y_test, y_pred)

        print("Pipeline executed and model saved as 'trained_face_recognition_model.joblib'")
    else:
        print("Data loading and preprocessing were unsuccessful. Please try again.")

if __name__ == "__main__":
    execute_pipeline()

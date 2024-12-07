import cv2
import numpy as np
import joblib
from data_preprocessing import flatten_images, normalize_data
from classification_module import predict_faces
from sklearn.decomposition import PCA

MODEL_PATH = 'trained_face_recognition_model.joblib'

# Load the trained classifier model
classifier = joblib.load(MODEL_PATH)

# Set up OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (h, w))

        # Flatten and normalize the face image
        face_flat = flatten_images(face_resized[np.newaxis, :])
        face_normalized = normalize_data(face_flat)

        # Perform PCA dynamically on each face, limit components based on the number of features
        n_components = min(100, face_normalized.shape[1])  # Ensure the number of components doesn't exceed the number of features
        pca = PCA(n_components=n_components)
        face_pca = pca.fit_transform(face_normalized)  # Perform PCA on the fly

        # Predict the label using the classifier
        prediction = predict_faces(classifier, face_pca)
        label = prediction[0]

        # Draw rectangle around the face and display the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognize_face(frame)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

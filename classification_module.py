from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_classifier(X_train, y_train, model_type='svm'):
    if model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
    elif model_type == 'logistic':
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)
        best_classifier = classifier
    return best_classifier

def predict_faces(classifier, X_test):
    return classifier.predict(X_test)

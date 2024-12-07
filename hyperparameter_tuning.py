from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(classifier, X_train, y_train, param_grid):
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

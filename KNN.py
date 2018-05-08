from imports import *

def knn(xTrain, yTrain, xTest, yTest):
    print("Hyperparameter Tuning!")
    gridClassifier = KNeighborsClassifier()
    nearestNeighbors = [1, 3, 5, 10]
    parameters = {'n_neighbors': nearestNeighbors}
    gridSearch = GridSearchCV(estimator=gridClassifier,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=5)
    gridSearch.fit(xTrain, yTrain.values.ravel())
    bestAccuracyMLP = gridSearch.best_score_
    bestParametersMLP = gridSearch.best_params_

    print("The best parameters for KNN model are :\n{}\n".format(bestParametersMLP))
    knnClassifier = KNeighborsClassifier(n_neighbors=1)
    knnClassifier.fit(xTrain, yTrain.values.ravel())
    yPredKNNTest = knnClassifier.predict(xTest)
    yPredKNNTrain = knnClassifier.predict(xTrain)
    print("KNN Evaluation :\n")
    print("Training Accuracy => {}".format(accuracy_score(yTrain, yPredKNNTrain) * 100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredKNNTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredKNNTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredKNNTest)))
    plt.scatter(yTest, yPredKNNTest)
  

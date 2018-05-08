from imports import *

# Decision Tree Classifier
def dt(xTrain, yTrain, xTest, yTest):
    print("Hyperparameter Tuning!")
    gridClassifier = DecisionTreeClassifier()
    depthList = [1,3,5,10]
    parameters = {'max_depth':depthList}
    gridSearch = GridSearchCV(estimator=gridClassifier,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=5)
    gridSearch.fit(xTrain, yTrain.values.ravel())
    bestAccuracyMLP = gridSearch.best_score_
    bestParametersMLP = gridSearch.best_params_

    print("The best parameters for Decision Tree model are :\n{}\n".format(bestParametersMLP))
    dtclassifier = DecisionTreeClassifier(max_depth=3)
    dtclassifier.fit(xTrain, yTrain.values.ravel())
    yPredictiondtTest = dtclassifier.predict(xTest)
    yPredictiondtTrain = dtclassifier.predict(xTrain)
    print("Decision Tree Evaluations :\n")
    print("Training Accuracy => {}".format(accuracy_score(yTrain, yPredictiondtTrain) * 100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredictiondtTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredictiondtTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredictiondtTest)))
    plt.scatter(yTest, yPredictiondtTest)
   

from imports import *

def logisticRegression(xTrain, yTrain, xTest, yTest):
    logRegClassifier = LogisticRegression(multi_class='ovr', random_state=0, C=3)
    logRegClassifier.fit(xTrain, yTrain.values.ravel())
    yPredLogTest = logRegClassifier.predict(xTest)
    yPredLogTrain = logRegClassifier.predict(xTrain)
    print("Logistic Regression Evaluation :\n")
    print("Testing Accuracy => {}".format(accuracy_score(yTest, yPredLogTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredLogTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredLogTest)))
    plt.scatter(yTest, yPredLogTest)

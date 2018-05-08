from imports import *
import preprocessing

def mlp(df, scaledData, targetVariable):

    print("SIT BACK AND RELAX! CROSS VALIDATION WOULD TAKE SOME TIME...")
    print("LOT OF STATEMENTS LOOKING LIKE ERRORS MIGHT POP UP...\n")

    # For MLP, we have to use scaled data
    scaledDF = preprocessing.scalingDataset(df)
    xTrain, xTest, yTrain, yTest = train_test_split(scaledData, targetVariable, test_size=0.2, random_state=0, shuffle=False)
    xTrain = xTrain.sort_index()
    xTest = xTest.sort_index()
    yTrain = yTrain.sort_index()
    yTest = yTest.sort_index()
    mlpClass = MLPClassifier()
    iterList = [200, 500, 1000]
    hiddenLayerList = [(100, 150), (100, 200), (100, 250)]
    parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'max_iter': iterList, 'hidden_layer_sizes': hiddenLayerList}
    gridSearch = GridSearchCV(estimator=mlpClass,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=10)
    gridSearch.fit(xTrain, yTrain.values.ravel())
    bestAccuracyMLP = gridSearch.best_score_
    bestParametersMLP = gridSearch.best_params_
    print("The best parameters for MLP model are :\n{}\n".format(bestParametersMLP))

    mlpClass = MLPClassifier(hidden_layer_sizes=(100, 200), alpha=0.1, max_iter=500)
    mlpClass.fit(xTrain, yTrain.values.ravel())
    yPredMLP = mlpClass.predict(xTest)
    print("Multilayer Perceptron Evaluation :\n")
    print("Training Accuracy => {}".format(bestAccuracyMLP*100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredMLP)*100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredMLP)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredMLP)))
    plt.scatter(yTest, yPredMLP)


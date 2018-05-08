from imports import *
import preprocessing

def kBestLogReg(filledData):
    bestDF = filledData.drop(columns=['Empathy'], axis=1)
    targetVariable = filledData['Empathy'].to_frame()
    selected = SelectKBest(score_func=f_classif, k=20)
    selectedFit = selected.fit(bestDF, targetVariable.values.ravel())
    selectedFitTransform = selectedFit.transform(bestDF)

    xTrainBestK, xTestBestK, yTrainBestK, yTestBestK = train_test_split(
        preprocessing.scalingDataset(selectedFitTransform),
        targetVariable,
        test_size=0.2,
        random_state=0)
    print("Cross Validating for best parameters..")
    print("This might take some time..\n")
    lr = LogisticRegression(multi_class='ovr')
    cList = [10, 100, 1000, 10000]
    solverList = ['lbfgs', 'sag', 'saga', 'newton-cg']
    maxIterList = [100, 1000, 10000]
    parameters = {'C': cList, 'solver': solverList, 'max_iter': maxIterList}
    gridSearch = GridSearchCV(estimator=lr,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=4)
    gridSearch.fit(xTrainBestK, yTrainBestK.values.ravel())
    bestAccuracyLogBestK = gridSearch.best_score_
    bestParametersLogBestK = gridSearch.best_params_
    print("The best parameters for Logistic Regression model are :\n{}\n".format(bestParametersLogBestK))
    # Best parameters : C:10, maxiter:100, solver:sag
    lr = LogisticRegression(C=10, max_iter=100, solver='lbfgs', multi_class='ovr', random_state=1)
    lr.fit(xTrainBestK, yTrainBestK.values.ravel())
    yPredLogBestTest = lr.predict(xTestBestK)
    bestKLogAcc = accuracy_score(yTestBestK, yPredLogBestTest)
    print("Logistic Regression using SelectKBest Method Evaluations :\n")
    print("Training Accuracy : {}".format(bestAccuracyLogBestK*100))
    print("Testing Accuracy  : {}\n".format(bestKLogAcc*100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTestBestK, yPredLogBestTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTestBestK, yPredLogBestTest)))
    plt.scatter(yTestBestK, yPredLogBestTest)

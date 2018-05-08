# Import all from imports
from imports import *
import preprocessing
import plotting
import dataImport
import decisionTree
import KNN
import MLP
import selectKBestlogReg
import logReg

if __name__== "__main__":
    
    print("**********************************************************************")
    print("EMPATHY PREDICTION USING LOGISTIC REGRESSION AND MULTILAYER PERCEPTRON")
    print("**********************************************************************")

    # Loading the dataset
    responsesData, columnsData = dataImport.importDataset()
    print("Datasets loaded!")
    print("**********************************************************************")

    # Collect the filled data
    print("Preprocessing data might take some time..")
    print("1) Missing values are being handled!")
    print("2) Categorical entries are getting converted to numeric values!")
    print("3) Dummy variables are being handled!")

    filledData = preprocessing.preprocessingDataset(responsesData)
    print("Done preprocessing data!")
    print("**********************************************************************")

    # Collect the scaled data if needed
    scaledData = preprocessing.scalingDataset(filledData)
    print("Now, data normalization is being done!")
    print("Done scaling of preprocessed data!")
    print("**********************************************************************")

    # Initializing target variable
    TV = "Empathy"
    targetVariable = filledData['Empathy'].to_frame()
    corrData = plotting.correlationFigure(scaledData, TV)
    corrData.sort_values(by="correlation_values", ascending=True)
    importantFeatures = corrData.tail(20)
    finalFeatures = pd.DataFrame()
    finalFeatures = importantFeatures
    finalColumnsList = []
    for x in finalFeatures['features']:
        finalColumnsList.append(x)

    df = pd.DataFrame()
    df = filledData[finalColumnsList[0]].to_frame()
    for x in range(1, len(finalColumnsList)):
        df = df.join(filledData[finalColumnsList[x]].to_frame())

    print("Feature Engineering is done!")
    print("Correlations are found out and top 20 features are chosen for modelling!")


    xTrain, xTest, yTrain, yTest = train_test_split(df, targetVariable, test_size=0.2, random_state=0)
    xTrain = xTrain.sort_index()
    xTest = xTest.sort_index()
    yTrain = yTrain.sort_index()
    yTest = yTest.sort_index()

    print("**********************************************************************")
    print("Ready for modelling! Please select the number from the below list.")
    print("1) Decision Tree Model.")
    print("2) KNN Model.")
    print("3) Logistic Regression Model.")
    print("4) Logistic Regression Model using 'SelectKBest' method of feature selection.")
    print("5) Multi-layer Perceptron Model.")

    i = True
    while i == True:
        userInput = input("Enter the number!")
        if int(userInput) == 1:
            decisionTree.dt(xTrain, yTrain, xTest, yTest)
        elif int(userInput) == 2:
            KNN.knn(xTrain, yTrain, xTest, yTest)
        elif int(userInput) == 3:
            logReg.logisticRegression(xTrain, yTrain, xTest, yTest)
        elif int(userInput) == 4:
            selectKBestlogReg.kBestLogReg(filledData)
        elif int(userInput) == 5:
            MLP.mlp(df, scaledData, targetVariable)
        else:
            print("Invalid Entry!")

        yninput = input("Would you like to continue exploring other models? (Y/N)")
        if yninput == "Y" or yninput == "y":
            i = True
        elif yninput == "N" or yninput == "n":
            i = False
            break
        else:
            print("Invalid Entry!")
            i = False
            break

    print("\nProject Done! Please have a look at the Jupyter Notebook for the learning curves!")

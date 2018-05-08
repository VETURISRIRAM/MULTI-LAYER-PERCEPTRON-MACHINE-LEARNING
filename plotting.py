# From imports import all
from imports import *

# Simple plot function
def correlationFigure(featureVariablesMain, targetVariable):
    # Calculate correlation
    #print(featureVariablesMain.columns)
    #print(targetVariable.values)
    def correlationCalculation(targetVariable, featureVariables, features):
        columns = [] # For maintaining the feature names
        values = [] # For maintaining the corr values of features with "Empathy"

        # Traverse through all the input features
        for x in features:
            if x is not None:
                columns.append(x) # Append the column name
                # Calculate the correlation
                c = np.corrcoef(featureVariables[x], featureVariables[targetVariable])
                absC = abs(c) # Absolute value because important values might miss
                values.append(absC[0,1])

        corrValues = pd.DataFrame()
        dataDict = {'features': columns, 'correlation_values': values}
        corrValues = pd.DataFrame(dataDict)
        # Sort the value by correlation values
        sortedCorrValues = corrValues.sort_values(by="correlation_values")

        # Plot the graph to show the features with their correlation values
        figure, ax = plt.subplots(figsize=(15, 45), squeeze=True)
        ax.set_title("Correlation Coefficients of Features")
        sns.barplot(x=sortedCorrValues.correlation_values, y=sortedCorrValues['features'], ax=ax)
        ax.set_ylabel("-----------Corr Coefficients--------->")


        plt.show()

        return sortedCorrValues

    # Make a list of columns
    columns = []
    for x in featureVariablesMain.columns:
        columns.append(x)
    # Remove "Empathy" from df
    columns.remove(targetVariable)

    # Compute correlations
    correlations = correlationCalculation(targetVariable, featureVariablesMain, columns)
    return correlations
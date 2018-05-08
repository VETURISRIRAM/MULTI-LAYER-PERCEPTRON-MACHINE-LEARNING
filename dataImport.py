from imports import *

# Import the datasets
def importDataset():
	dataset = pd.read_csv('responses.csv')
	columns = pd.read_csv('columns.csv')
	return dataset, columns
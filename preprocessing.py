# Import all from imports
from imports import *

# Pre processing the data set provided
def preprocessingDataset(dataset):

    # Define imp from Imputer class for missing values
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    #### Preprocessing the Dataset
    music = dataset.iloc[:, 0:19]
    movies = dataset.iloc[:, 19:31]
    phobias = dataset.iloc[:, 63:73]
    interests = dataset.iloc[:, 31:63]
    health = dataset.iloc[:, 73:76]
    personal = dataset.iloc[:, 76:133]
    information = dataset.iloc[:, 140:150]
    expenditure = dataset.iloc[:, 133:140]

    """
    print(music)
    print(movies)
    print(phobias)
    print(interests)
    print(health)
    print(personal)
    print(information)
    print(spendings)
    """
    # Processing the personal
    for x in personal["Lying"]:
        if x == "never":
            personal.replace(x, 1.0, inplace=True)
        elif x == "only to avoid hurting someone":
            personal.replace(x, 2.0, inplace=True)
        elif x == "sometimes":
            personal.replace(x, 3.0, inplace=True)
        elif x == "everytime it suits me":
            personal.replace(x, 4.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    for x in personal["Punctuality"]:
        if x == "i am often early":
            personal.replace(x, 3.0, inplace=True)
        elif x == "i am always on time":
            personal.replace(x, 2.0, inplace=True)
        elif x == "i am often running late":
            personal.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    for x in personal["Internet usage"]:
        if x == "most of the day":
            personal.replace(x, 4.0, inplace=True)
        elif x == "few hours a day":
            personal.replace(x, 3.0, inplace=True)
        elif x == "less than an hour a day":
            personal.replace(x, 2.0, inplace=True)
        elif x == "no time at all":
            personal.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    # Replace strings with numpy NaNs
    personal = personal.replace("NaN", np.nan)
    personal = personal.replace("nan", np.nan)

    # Replace missing values with most frequent values
    imp.fit(personal)
    personal_data = imp.transform(personal)

    d = personal_data[:, :]
    ind = []
    for x in range(len(personal_data)):
        ind.append(x)
    c = personal.columns.tolist()
    personal = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing the health
    for x in health["Smoking"]:
        if x == "current smoker":
            health.replace(x, 1.0, inplace=True)
        elif x == "former smoker":
            health.replace(x, 2.0, inplace=True)
        elif x == "tried smoking":
            health.replace(x, 3.0, inplace=True)
        elif x == "never smoked":
            health.replace(x, 4.0, inplace=True)
        elif x == "Nan":
            health.replace(x, np.nan, inplace=True)
        elif x == "nan":
            health.replace(x, np.nan, inplace=True)

    for x in health["Alcohol"]:
        if x == "drink a lot":
            health.replace(x, 1.0, inplace=True)
        elif x == "social drinker":
            health.replace(x, 2.0, inplace=True)
        elif x == "never":
            health.replace(x, 3.0, inplace=True)
        elif x == "Nan":
            health.replace(x, np.nan, inplace=True)
        elif x == "nan":
            health.replace(x, np.nan, inplace=True)

    # Replace strings with numpy NaNs
    health = health.replace("NaN", np.nan)
    health = health.replace("nan", np.nan)

    # Replace missing values with most frequent values
    imp.fit(health)
    healthData = imp.transform(health)
    d = healthData[:, :]
    ind = []
    for x in range(len(healthData)):
        ind.append(x)
    c = health.columns.tolist()
    health = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing the information
    for x in information["Gender"]:
        if x == "female":
            information.replace(x, 2.0, inplace=True)
        elif x == "male":
            information.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Left - right handed"]:
        if x == "right handed":
            information.replace(x, 1.0, inplace=True)
        elif x == "left handed":
            information.replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Education"]:
        if x == "doctorate degree":
            information.replace(x, 6.0, inplace=True)
        elif x == "masters degree":
            information.replace(x, 5.0, inplace=True)
        elif x == "college/bachelor degree":
            information.replace(x, 4.0, inplace=True)
        elif x == "secondary school":
            information.replace(x, 3.0, inplace=True)
        elif x == "primary school":
            information.replace(x, 2.0, inplace=True)
        elif x == "currently a primary school pupil":
            information.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Only child"]:
        if x == "yes":
            information.replace(x, 1.0, inplace=True)
        elif x == "no":
            information.replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Village - town"]:
        if x == "village":
            information["Village - town"].replace(x, 1.0, inplace=True)
        elif x == "city":
            information["Village - town"].replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["House - block of flats"]:
        if x == "block of flats":
            information["House - block of flats"].replace(x, 1, inplace=True)
        elif x == "house/bungalow":
            information["House - block of flats"].replace(x, 2, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    information = information.replace("nan", np.nan)
    information = information.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(information)
    informationData = imp.transform(information)
    d = informationData[:, :]
    ind = []
    for x in range(len(informationData)):
        ind.append(x)
    c = information.columns.tolist()
    information = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing music
    music = music.replace("nan", np.nan)
    music = music.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(music)
    musicData = imp.transform(music)
    d = musicData[:, :]
    ind = []
    for x in range(len(musicData)):
        ind.append(x)
    c = music.columns.tolist()
    music = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing movies
    movies = movies.replace("nan", np.nan)
    movies = movies.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(movies)
    moviesData = imp.transform(movies)
    d = moviesData[:, :]
    ind = []
    for x in range(len(moviesData)):
        ind.append(x)
    c = movies.columns.tolist()
    movies = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing phobias
    phobias = phobias.replace("nan", np.nan)
    phobias = phobias.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(phobias)
    phobiasData = imp.transform(phobias)
    d = phobiasData[:, :]
    ind = []
    for x in range(len(phobiasData)):
        ind.append(x)
    c = phobias.columns.tolist()
    phobias = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing interests
    interests = interests.replace("nan", np.nan)
    interests = interests.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(interests)
    interestsData = imp.transform(interests)
    d = interestsData[:, :]
    ind = []
    for x in range(len(interestsData)):
        ind.append(x)
    c = interests.columns.tolist()
    interests = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing spendings
    expenditure = expenditure.replace("nan", np.nan)
    expenditure = expenditure.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(expenditure)
    expenditureData = imp.transform(expenditure)
    d = expenditureData[:, :]
    ind = []
    for x in range(len(expenditureData)):
        ind.append(x)
    c = expenditure.columns.tolist()
    expenditure = pd.DataFrame(data=d, index=ind, columns=c)

    # Joining all the processed sections
    joinedDatasets = music.join(movies.join(phobias.join(interests.join(health.join(personal.join(information.join(expenditure)))))))

    return joinedDatasets

# Scale the data if necessary
def scalingDataset(dataset):
    # Scaling the dataset
    scaler = StandardScaler()
    scaledDataarray = scaler.fit_transform(dataset)
    if type(dataset) is np.ndarray:
        return scaledDataarray
    else:
        d = scaledDataarray[:, :]
        ind = []
        for x in range(len(dataset)):
            ind.append(x)
        c = dataset.columns.tolist()
        scaledData = pd.DataFrame(data=d, index=ind, columns=c)
        return scaledData
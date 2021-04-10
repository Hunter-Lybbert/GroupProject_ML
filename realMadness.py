from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split as split
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def shuffle_data(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    newX = [d[0] for d in data]
    newy = [d[1] for d in data]
    return np.array(newX), np.array(newy)

files = ["data/NCAA_Season_Stats_" + str(i) + ".csv" for i in range(2000, 2022)]

MSEs = []
scores = []

for n, f in enumerate(files):
    if f == 'data/NCAA_Season_Stats_2020.csv':
        continue
    dataFrame = pd.read_csv(f)
    npXy = np.array(dataFrame)

    X = npXy[:, 1:-1]
    y = npXy[:, -1]
    # Normalize data
    X = ((X - np.min(X, 0)) / (np.max(X, 0) - np.min(X, 0) + .0001))

    shuffleX, shuffley = shuffle_data(X, y)

    trainX = shuffleX[:int(.8 * len(X)), :]
    trainy = shuffley[:int(.8 * len(y))]

    #validX = trainX[:int(.15 * len(X)), :]
    #validy = trainy[:int(.15 * len(y))]

    testX = shuffleX[int(.8 * len(X)):, :]
    testy = shuffley[int(.8 * len(y)):]

    mMadness = MLP(random_state=0, hidden_layer_sizes=(15, 15), activation='identity',
                   alpha=.0001, learning_rate_init=.2, max_iter=30).fit(trainX, trainy)
    lossCurve = mMadness.loss_curve_
    predictions = mMadness.predict(testX)

    MSE = sum([(int(predictions[i]) - testy[i]) ** 2 for i in range(len(predictions))]) / len(predictions)

    score = mMadness.score(testX, testy)
    MSEs.append(MSE)
    scores.append(score)

    print(f)
    print("MSE: " + str(MSE))
    print("Score: " + str(score))
    print()
    plt.plot(range(len(lossCurve)), lossCurve, label=str(n + 2000))
avgMSE = sum(MSEs)/len(MSEs)
avgScore = sum(scores)/len(scores)
print("Average MSE: " + str(avgMSE))
print("Average Score: " + str(avgScore))
print()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss function measured yearly")
plt.show()

def read_data(start, end):
    '''
    Reads all data and places it in a numpy array
    Start: first year of data
    end: last year of data

    Returns: numpy array of all data'''

    # read in data and place it into a single dataframe
    df = None
    for i in range(start, end + 1):
        if i != 2020:
            tmp_df = pd.read_csv('data/NCAA_Season_Stats_{}.csv'.format(i))
            if df is None:
                df = tmp_df.copy()
            else:
                df = pd.concat([df, tmp_df])

    # replace the remaining Nan values with the averages of their columns
    for n, i in enumerate(sum(df.isnull().values)):
        if i != 0:
            mean = df.iloc[:, n].astype('float64').mean()
            df.iloc[:, n].fillna(value=mean, inplace=True)

    return np.array(df)

data = np.array(read_data(1993,2021))

#data.shape
X = data[:,1:-1].astype('float64')
y=data[:,-1].astype(int)
X, y = shuffle_data(X, y)


X_train, X_test, y_train, y_test = split(X,y,test_size=.3, random_state=8)

mMadness = MLP(random_state=0, hidden_layer_sizes=(15, 15), activation='identity',
                   alpha=.0001, learning_rate_init=.2, max_iter=30).fit(X_train, y_train)

lossCurve = mMadness.loss_curve_
plt.plot(range(len(lossCurve)), lossCurve)

predictions = mMadness.predict(X_test)

MSEStacked = sum([(int(predictions[i]) - y_test[i]) ** 2 for i in range(len(predictions))]) / len(predictions)

scoreStacked = mMadness.score(X_test, y_test)

print("All Stacked:")
print("MSE: " + str(MSEStacked))
print("Score: " + str(scoreStacked))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss function with all data from 2000-2021")
plt.show()
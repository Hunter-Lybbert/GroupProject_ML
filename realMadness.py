from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split as split
import random
import numpy as np
import pandas as pd

def shuffle_data(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    newX = [d[0] for d in data]
    newy = [d[1] for d in data]
    return np.array(newX), np.array(newy)

files = ["data/NCAA_Season_Stats_" + str(i) + ".csv" for i in range(2000, 2022)]

MSEs = []
scores = []

for f in files:
    if f == 'data/NCAA_Season_Stats_2020.csv':
        continue
    dataFrame = pd.read_csv(f)
    # print(dataFrame)
    npXy = np.array(dataFrame)

    X = npXy[:, 1:-1]
    y = npXy[:, -1]

    shuffleX, shuffley = shuffle_data(X, y)

    trainX = shuffleX[:int(.8 * len(X)), :]
    trainy = shuffley[:int(.8 * len(y))]

    #validX = trainX[:int(.15 * len(X)), :]
    #validy = trainy[:int(.15 * len(y))]

    testX = shuffleX[int(.8 * len(X)):, :]
    testy = shuffley[int(.8 * len(y)):]

    mMadness = MLP(random_state=0, hidden_layer_sizes=(15, 15), activation='identity',
                   alpha=.0001, learning_rate_init=.2, max_iter=120).fit(trainX, trainy)

    predictions = mMadness.predict(testX)

    MSE = sum([(int(predictions[i]) - testy[i]) ** 2 for i in range(len(predictions))]) / len(predictions)

    score = mMadness.score(testX, testy)
    MSEs.append(MSE)
    scores.append(score)

    print(f)
    print("MSE: " + str(MSE))
    print("Score: " + str(score))
    print()

print("Average MSE: " + str(sum(MSEs)/len(MSEs)))
print("Average Score: " + str(sum(scores)/len(scores)))

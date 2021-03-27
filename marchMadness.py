from sklearn.neural_network import MLPClassifier as MLP
import random
import numpy as np
import pandas as pd

file = open("472_Fake_data.csv")
# npXy = file.read().strip().split('\n')
dataFrame = pd.read_csv("472_Fake_data.csv", header=None)
#print(dataFrame)
npXy = np.array(dataFrame)

for i in range(10):
    npXy = np.vstack((npXy, npXy))

m, n = npXy.shape

for col in range(n - 1):
    currCol = npXy[:, col]
    colMin = np.min(currCol)
    colMax = np.max(currCol)
    adjustLength = (colMax - colMin) * .5
    for row in range(m):
        randAdjust = random.uniform(-adjustLength, adjustLength)
        npXy[row, col] += randAdjust


print(npXy)
print(npXy.shape)


X = npXy[:, :-1]
y = npXy[:, -1]

# print(X)
# print(y)

def shuffle_data(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    newX = [d[0] for d in data]
    newy = [d[1] for d in data]
    return np.array(newX), np.array(newy)

shuffleX, shuffley = shuffle_data(X, y)

trainX = shuffleX[:int(.8*len(X)), :]
trainy = shuffley[:int(.8*len(y))]

validX = trainX[:int(.15 * len(X)), :]
validy = trainy[:int(.15 * len(y))]

testX = shuffleX[int(.8 * len(X)):, :]
testy = shuffley[int(.8 * len(y)):]

def searchHyperParams(trainX, trainy, testX, testy):
    bestAccuracy = 0.0
    bestParams = None
    for node in [5*i for i in range(1, 6)]:
        for func in ['identity', 'logistic', 'tanh', 'relu']:
            for alpha in [.0001*i for i in range(1, 6)]:
                for lr in [.01*i for i in range(1, 40)]:
                    iris = MLP(random_state=0, hidden_layer_sizes=(node, node), activation=func,
                               alpha=alpha, learning_rate_init=lr, max_iter=120)
                    fitted = iris.fit(trainX, trainy)
                    score = fitted.score(testX, testy)
                    if score > bestAccuracy:
                        bestAccuracy = score
                        bestParams = [node, func, alpha, lr]
                        print("Score: " + str(score))
                        print(bestParams)
                        if score == 1.0:
                            print("Best accuracy: " + str(bestAccuracy))
                            print("Nodes/Layers: " + str(bestParams[0]))
                            print("Activation function: " + str(bestParams[1]))
                            print("Regularization alpha: " + str(bestParams[2]))
                            print("Learning rate: " + str(bestParams[3]))
                            return
    print("Best accuracy: " + str(bestAccuracy))
    print("Nodes/Layers: " + str(bestParams[0]))
    print("Activation function: " + str(bestParams[1]))
    print("Regularization alpha: " + str(bestParams[2]))
    print("Learning rate: " + str(bestParams[3]))

# searchHyperParams(trainX, trainy, testX, testy)

mMadness = MLP(random_state=0, hidden_layer_sizes=(15, 15), activation='identity',
                               alpha=.0001, learning_rate_init=.2, max_iter=120).fit(trainX, trainy)

predictions = mMadness.predict(testX)

MSE = sum([(predictions[i] - testy[i])**2 for i in range(len(predictions))]) / len(predictions)
print(MSE)
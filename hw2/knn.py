import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Model Selection for KNN
"""

def main():
    K = range(1, 52, 2)

    trainResult = []
    leaveOneOut = []
    testResult = []

    train, test = getData()

    for k in K:
        print("--> " + str(k) + "th nearest neighbor...")
        trainResult.append(knnTraining(train, train, k))
        testResult.append(knnTraining(train, test, k))
        leaveOneOut.append(knnLeaveOneOut(train, k))

    printResult(trainResult, testResult, leaveOneOut)
    plot(K, trainResult, testResult, leaveOneOut)


def knnLeaveOneOut(train, k):
    # Leave-one-out cross validation
    errors = []
    correct = 0
    nn = [getNN(train, row) for row in train['X']]

    for i, neighbors in enumerate(nn):
        knnResults = sum([dist.values()[1] for dist in neighbors[1:k + 1]])
        test = neighbors[0]['result']

        if (knnResults > 0 and test == 1) or (knnResults < 0 and test == -1):
            correct += 1

    accuracy = float(correct) / len(train['X'])
    return {'accuracy': accuracy, 'errors': len(train['X']) - correct}


def knnTraining(train, test, k):
    # Kth NN traning with test data
    correct = 0
    nn = [getNN(train, row) for row in test['X']]

    for i, neighbors in enumerate(nn):
        knnResults = sum([dist.values()[1] for dist in neighbors[:k]])

        if (knnResults > 0 and test['Y'][i] == 1) or \
                (knnResults < 0 and test['Y'][i] == -1):
            correct += 1

    accuracy = float(correct) / len(test['X'])
    return {'accuracy': accuracy, 'errors': len(test['X']) - correct}


def getNN(train, row):
    # Get all neighbors and sort them by nearness
    neighbors = []
    for i, val in enumerate(train['X']):

        distance = math.sqrt(sum((val - row) ** 2))
        neighbors.append({
            'result': train['Y'][i],
            'distance': distance,
        })

    neighbors = sorted(neighbors, key=lambda x: x['distance'])
    return neighbors


def plot(K, train, test, loo):
    # plot accuracy and errors
    trainAccuracy = [d['accuracy'] for d in train]
    testAccuracy = [d['accuracy'] for d in test]
    looAccuracy = [d['accuracy'] for d in loo]
    trainErrors = [d['errors'] for d in train]
    testErrors = [d['errors'] for d in test]
    looErrors = [d['errors'] for d in loo]

    plt.plot(K, trainAccuracy, marker='o', label="training data accuracy")
    plt.plot(K, testAccuracy, marker='o', label="testing data accuracy")
    plt.plot(K, looAccuracy, marker='o', label="leave-one-out cross validation accuracy")
    plt.xlabel("# of K")
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('knnAccuracy.png')
    plt.clf()

    axes = plt.gca()
    axes.set_ylim([0, 30])
    plt.plot(K, trainErrors, marker='o', label="training data error")
    plt.plot(K, testErrors, marker='o', label="testing data error")
    plt.plot(K, looErrors, marker='o', label="leave-one-out cross validation error")
    plt.xlabel("# of K")
    plt.ylabel('# of Errors')
    plt.legend()
    plt.savefig('knnErrors.png')

def printResult(train, test, loo):
    print("------> Training data errors")
    print([d['errors'] for d in train])
    print("------> Testing data errors")
    print([d['errors'] for d in test])
    print("------> Leave-one-out cross validation data errors")
    print([d['errors'] for d in loo])
    print("")
    print("------> Training data accuracy")
    print([d['accuracy'] for d in train])
    print("------> Testing data accuracy")
    print([d['accuracy'] for d in test])
    print("------> Leave-one-out cross validation accuracy")
    print([d['accuracy'] for d in loo])


def getData():
    # get data extract data into varables
    trainData = loadFile("./data/knn_train.csv")
    testData = loadFile("./data/knn_test.csv")

    trainX = normalize(trainData[:, 1:])
    trainY = trainData[:, 0]
    testX = normalize(testData[:, 1:])
    testY = testData[:, 0]

    train = {
        'X': trainX,
        'Y': trainY,
    }
    test = {
        'X': testX,
        'Y': testY,
    }

    return train, test


def normalize(X):
    # normalize data
    min = X.min(axis=0)
    max = X.max(axis=0)
    X = (X - min) / (max - min)
    return X


def loadFile(path):
    # load data from file and return data matrix
    file = open(path, "r")
    data = np.genfromtxt(file, delimiter=",")
    return data


if __name__ == "__main__":
	main()

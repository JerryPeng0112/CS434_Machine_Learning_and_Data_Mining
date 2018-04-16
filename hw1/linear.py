import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

"""
Linear Regression Training
"""

def main():
    # Linear regression with dummy variable
    print "\n====== With Dummy Variable ======\n"
    train(True)
    # Linear regression without dummy variable
    print "\n====== Without Dummy Variable ======\n"
    train(False)

    print "\n====== Training with normal distributed random data ======\n"
    normDataTrain()

def train(dummyVar):
    inputData = loadData("./data/housing_train.txt")

    # load data into matrix X and Y
    if dummyVar:
        X = np.insert(inputData[:, 0:-1], 0, 1, axis=1)
    else:
        X = inputData[:, 0:-1]

    Y = inputData[:, -1]

    w = calcWeight(X, Y)

    print "Weight vector:"
    print w

    calcASE(w, dummyVar)

def normDataTrain():
    trainASE = []
    testASE = []
    dRange = 52

    for d in range(0, dRange, 2):
        inputData = loadData("./data/housing_train.txt")
        X = np.insert(inputData[:, 0:-1], 0, 1, axis=1)
        Y = inputData[:, -1]
        n = X.shape[0]

        inputData = loadData("./data/housing_test.txt")
        testX = np.insert(inputData[:, 0:-1], 0, 1, axis=1)
        testY = inputData[:, -1]
        testN = testX.shape[0]

        # add random data from normal distribution
        for i in range(d):
            mu = np.random.rand() * 1000 + 1000
            sigma = np.random.rand() * 1000 + 1000
            normalData = np.array(np.random.normal(mu, sigma, n))

            X = np.insert(X, X.shape[1], normalData, axis=1)
            normalTestingData = np.random.normal(mu, sigma, testN)
            testX = np.insert(testX, testX.shape[1], normalTestingData, axis=1)

        w = calcWeight(X, Y)
        trainASE.append(calcASEwithNorm(X, Y, w))
        testASE.append(calcASEwithNorm(testX, testY, w))

    print "Training ASE:"
    print trainASE
    print "Testing ASE:"
    print testASE

    plotASE(trainASE, testASE, range(0, dRange, 2))


def calcASE(w, dummyVar):
    # calculate ASE for training & test data
    files = ["./data/housing_train.txt", "./data/housing_test.txt"]
    names = ["Training data ASE: ", "Testing data ASE: "]
    w = np.transpose(w)

    for idx, f in enumerate(files):
        inputData = loadData(f)

        if dummyVar:
            X = np.insert(inputData[:, 0:-1], 0, 1, axis=1)
        else:
            X = inputData[:, 0:-1]
        Y = inputData[:, -1]

        result = np.dot(X, w)
        squaredDiffs = (result - Y) ** 2
        SSE = sum(squaredDiffs)
        ASE = SSE / len(squaredDiffs)

        print names[idx]
        print ASE

def calcASEwithNorm(X, Y, w):
    result = np.dot(X, w)
    squaredDiffs = (result - Y) ** 2
    SSE = sum(squaredDiffs)
    ASE = SSE / len(squaredDiffs)
    return ASE

def plotASE(trainASE, testASE, dRange):
    # plot training error
	plt.plot(dRange, trainASE, label="training data")
	plt.xlabel('d Value')
	plt.ylabel('ASE')
	#plot testing error
	plt.plot(dRange, testASE, label="testing data")
	plt.xlabel('d Value')
	plt.ylabel('ASE')
	plt.savefig('NormDataASE.png')


def loadData(path):
    # load data from file and return data matrix
    file = open(path, "r")
    data = np.genfromtxt(file, delimiter=" ")
    return data

def calcWeight(X, Y):
    # calculate weight vector
    Xt = np.transpose(X)
    a = np.linalg.inv(np.matmul(Xt, X))
    b = np.matmul(Xt, Y)
    w = np.matmul(a, b)
    return w


if __name__ == "__main__":
	main()

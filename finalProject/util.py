import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def loadGeneralTrainData():
    # Load general training data
    path = "./data/generalTrainingData/"
    subjectFilename = ["subject1.csv", "subject4.csv", "subject6.csv", "subject9.csv"]
    indexFilename = ["index1.csv", "index4.csv", "index6.csv", "index9.csv"]

    data = []
    indexes = []

    for filename in subjectFilename:
        data.append(loadData(path + filename, True))

    for filename in indexFilename:
        indexes.append(loadData(path + filename))

    return data, indexes


def loadGeneralTestData():
    # Load general testing data set
    path = "./data/testData/finalTest/general_test_instances.csv"
    return loadData(path)[:, 7:]


def loadSingleTrainData():
    # Load single traning data
    path = "./data/singleTrainingData/"
    data2 = loadData(path + "subject2_training.csv", True)
    data7 = loadData(path + "subject7_training.csv", True)
    index2 = loadData(path + "index2_training.csv")
    index7 = loadData(path + "index7_training.csv")

    return data2, data7, index2, index7


def loadSingleTestData():
    # Load single testing data set
    path = ["./data/testData/finalTest/subject2_instances.csv",
        "./data/testData/finalTest/subject7_instances.csv"]
    return loadData(path[0])[:, 7:], loadData(path[1])[:, 7:]


def generalTransform(data, indexes):
    # Transform data for general training set
    Xs, Ys = [], []

    for subjectIdx in range(4):
        indexList = indexes[subjectIdx]
        subjectData = data[subjectIdx]
        X, Y = transform(subjectData, indexList)

        Xs.append(X)
        Ys.append(Y)

    return Xs, Ys


def singleTransform(data2, data7, index2, index7):
    # Transform data for single training set
    X2, Y2 = transform(data2, index2)
    X7, Y7 = transform(data7, index7)
    return X2, X7, Y2, Y7


def transform(subjectData, indexList):
    # utility function to transform data into 63 feature instances
    X, Y = [], []

    for idx in range(0, len(indexList) - 6):
        if (indexList[idx] == indexList[idx + 6] - 6):

            instance = subjectData[idx: idx + 7].flatten('F')
            X.append(instance[7:63])
            Y.append(instance[-1])

    return np.array(X), np.array(Y)


def splitTrainTest(Xs, Ys, idx):
    # lambda function for concatenate numpy arrays
    f = lambda a, b: np.concatenate((a, b), axis=0)

    # split the training and testing data set
    X = Xs[ :idx] + Xs[idx+1:]
    Y = Ys[:idx] + Ys[idx+1:]
    X = reduce(f, X)
    Y = reduce(f, Y)
    Xt = Xs[idx]
    Yt = Ys[idx]

    return X, Y, Xt, Yt


def sampling(X, Y):
    # Sample more positive data
    posSamples = []

    for idx, val in enumerate(Y):
        if val == 1:
            posSamples.append(X[idx])

    posSamples = np.array(posSamples)

    numSamples = X.shape[0] - posSamples.shape[0]
    mean =  np.mean(posSamples, axis=0)
    std = np.std(posSamples, axis=0)

    randomSamples = np.random.normal(mean, std, (numSamples, X.shape[1]))
    X = np.concatenate((X, randomSamples), axis=0)

    ones = np.ones(numSamples).reshape(numSamples)
    Y = np.concatenate((Y, ones), axis=0)

    return X, Y


def pca(X, Xt):
    # function for principle component analysis
    pca = PCA(.95)
    pca.fit(X)
    pcaX = pca.transform(X)
    pcaXt = pca.transform(Xt)

    return pcaX, pcaXt


def scaler(X, Xt):
    # function for preprocessing
    scaler = StandardScaler()
    scaler.fit(X)
    stdX = scaler.transform(X)
    stdXt = scaler.transform(Xt)

    return stdX, stdXt


def loadData(path, convertDate=False):
    # load data from file and return data matrix
    file = open(path, "r")

    if convertDate:
        dateStr2Hour = lambda x: float(datetime.strptime(x.decode("utf-8"), '%Y-%m-%dT%H:%M:%SZ').hour)
        data = np.genfromtxt(file, delimiter=",", converters={0: dateStr2Hour})

    else:
        data = np.genfromtxt(file, delimiter=",")

    return data


def printAccuracy(YPredict, Yt, title):
    #print(YPredict)
    # Convert YPredict to binary values, and count the number of 1s
    f = lambda a: 0 if a < 0.5 else 1
    YPredict = map(f, YPredict)
    count0 = YPredict.count(0)
    count1 = YPredict.count(1)

    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(Yt) - 1):
        if YPredict[i] == 1 and Yt[i] == 1:
            TP = TP + 1
        elif YPredict[i] == 1 and Yt[i] == 0:
            FP = FP + 1
        elif YPredict[i] == 0 and Yt[i] == 1:
            FN = FN + 1
        else:
            TN = TN + 1

    print("\n==> {0}".format(title))
    print("# of 0: {0}, # of 1: {1}".format(str(count0), str(count1)))
    print("TP: {0}, FP: {1}, FN: {2}, TN: {3}".format(str(TP), str(FP), str(FN), str(TN)))


def reduceProb(pred, prob):
    # Remove useles value in package generated probability
    newProb = []

    for idx, val in enumerate(pred):
        if val == 0:
            newProb.append(prob[idx][0])
        else:
            newProb.append(prob[idx][1])

    return np.array(newProb)


def savePred(YPredict, YProb, filename):
    # Save prediction files
    f1 = lambda a: 0 if a < 0 else (1 if a > 1 else a)
    f2 = lambda a: 0 if a < 0.5 else 1
    proba = map(f1, YProb)
    YPredict = map(f2, YPredict)
    output = np.vstack((proba, YPredict)).T
    np.savetxt(filename, output, fmt='%0.12f,%i', delimiter=',')


def saveGroundTruth(Y, filename):
    # Save ground truth to csv file
    np.savetxt(filename, Y.astype(int), fmt='%i', delimiter=' ')

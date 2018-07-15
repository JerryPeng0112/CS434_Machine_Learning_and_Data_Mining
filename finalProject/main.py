import numpy as np

import util
from regression import linearRegression, logisticRegression
from classifier import sgdClassify, perceptron, paClassify, nbClassify, svm
from knnTrees import kdTree, decisionTree
from mlp import mlp
from cluster import kmean

"""
main function
"""

def main():
    #generalCVTrain()
    generalTrain()
    singleTrain()


def generalCVTrain():
    # General Training with cross validation
    data, indexes = util.loadGeneralTrainData()
    Xs, Ys = util.generalTransform(data, indexes)
    subjectIndexes = [1, 4, 6, 9]

    print("\n--> General Training Data Set Shape")
    for i in range(4):
        # used to output ground truth files
        #util.saveGroundTruth(Ys[i], "groundTruth{0}.csv".format(subjectIndexes[i]))

        print("Subject {0}. X: {1}, Y: {2}, # of 1: {3}".format(\
        subjectIndexes[i], Xs[i].shape, Ys[i].shape, Ys[i].tolist().count(1)))

    for i in range(4):
        print("\n=========== Subject {0} As Validation Set ===========".format(subjectIndexes[i]))

        X, Y, Xt, Yt = util.splitTrainTest(Xs, Ys, i)
        X, Y = util.sampling(X, Y)
        pcaX, pcaXt = util.pca(X, Xt)
        stdX, stdXt = util.scaler(pcaX, pcaXt)

        sampleTraining(stdX, Y, stdXt, Yt)


def generalTrain():
    # General training with final final testing set
    filename = "general_pred"
    data, indexes = util.loadGeneralTrainData()
    Xt = util.loadGeneralTestData()
    Xs, Ys = util.generalTransform(data, indexes)

    f = lambda a, b: np.concatenate((a, b), axis=0)
    X = reduce(f, Xs)
    Y = reduce(f, Ys)

    X, Y = util.sampling(X, Y)
    pcaX, pcaXt = util.pca(X, Xt)
    stdX, stdXt = util.scaler(pcaX, pcaXt)

    training(stdX, Y, stdXt, filename)



def singleTrain():
    # Single trainings with final single testing set
    filename = ['individual1_pred', 'individual2_pred']
    data2, data7, index2, index7 = util.loadSingleTrainData()
    Xt2, Xt7 = util.loadSingleTestData()
    X2, X7, Y2, Y7 = util.singleTransform(data2, data7, index2, index7)

    X2, Y2 = util.sampling(X2, Y2)
    X7, Y7 = util.sampling(X7, Y7)

    pcaX2, pcaXt2 = util.pca(X2, Xt2)
    stdX2, stdXt2 = util.scaler(pcaX2, pcaXt2)
    pcaX7, pcaXt7 = util.pca(X7, Xt7)
    stdX7, stdXt7 = util.scaler(pcaX7, pcaXt7)

    training(stdX2, Y2, stdXt2, filename[0])
    training(stdX7, Y7, stdXt7, filename[1])



def sampleTraining(X, Y, Xt, Yt=None):
    # Function used for testing various model of training

    # Class weight
    class_weight = {0:0.6, 1:0.4}

    linearRegression(X, Y, Xt, Yt)
    logisticRegression(X, Y, Xt, Yt, class_weight)
    sgdClassify(X, Y, Xt, Yt, class_weight)
    #perceptron(X, Y, Xt, Yt, class_weight)
    #paClassify(X, Y, Xt, Yt, class_weight)
    #nbClassify(X, Y, Xt, Yt)
    #svm(X, Y, Xt, Yt, class_weight)
    #kdTree(X, Y, Xt, Yt)
    #decisionTree(X, Y, Xt, Yt)
    #mlp(X, Y, Xt, Yt)
    #kmean(X, Y, Xt, Yt)


def training(X, Y, Xt, filename):
    # Function actually used to generate predictions for final test sets
    # Class weight
    class_weight = {0:0.6, 1:0.4}
    YPredict1 = linearRegression(X, Y, Xt, None)
    YPredict2, YProb2 = logisticRegression(X, Y, Xt, None, class_weight)
    YPredict3, YProb3 = sgdClassify(X, Y, Xt, None, class_weight)

    util.savePred(YPredict1, YPredict1, "{0}1.csv".format(filename))
    util.savePred(YPredict2, YProb2, "{0}2.csv".format(filename))
    util.savePred(YPredict3, YProb2, "{0}3.csv".format(filename))


if __name__ == "__main__":
    main()

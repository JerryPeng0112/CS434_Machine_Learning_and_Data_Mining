import numpy as np
from sklearn.linear_model import (
    SGDClassifier, Perceptron, PassiveAggressiveClassifier
    )
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from util import printAccuracy, reduceProb

def sgdClassify(X, Y, Xt, Yt, class_weight):

    title = "SGD Classifier"

    classifier = SGDClassifier(loss="log", alpha=0.01, class_weight=class_weight)
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)
    YProb = reduceProb(YPredict, classifier.predict_proba(Xt))

    if Yt is not None:
        printAccuracy(YPredict, Yt, title)
    else:
        return YPredict, YProb


def perceptron(X, Y, Xt, Yt, class_weight):

    title = "Perceptron"

    classifier = Perceptron(n_iter=10, class_weight=class_weight)
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)


def paClassify(X, Y, Xt, Yt, class_weight):

    title = "Passive Aggressive Classifier"

    classifier = PassiveAggressiveClassifier(n_iter=10, class_weight=class_weight)
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)


def nbClassify(X, Y, Xt, Yt):

    title = "Gaussian Naive Bayes Classifier"

    classifier = GaussianNB()
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)


def svm(X, Y, Xt, Yt, class_weight):

    title = "Support Vector Machine"

    classifier = SVC(class_weight=class_weight)
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)

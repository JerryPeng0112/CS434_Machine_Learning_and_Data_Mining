from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from util import printAccuracy

def kdTree(X, Y, Xt, Yt):

    title = "K-th Nearest Neighbors"

    classifier = KNeighborsClassifier()
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)


def decisionTree(X, Y, Xt, Yt):

    title = "Decision Tree"

    classifier = DecisionTreeClassifier()
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)

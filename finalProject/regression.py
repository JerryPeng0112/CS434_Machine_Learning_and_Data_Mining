from sklearn.linear_model import LinearRegression, LogisticRegression
from util import printAccuracy, reduceProb

def logisticRegression(X, Y, Xt, Yt, class_weight):

    title = "Logistic Regression"

    lr = LogisticRegression(class_weight=class_weight)
    lr.fit(X, Y)
    YPredict = lr.predict(Xt)
    YProb = reduceProb(YPredict, lr.predict_proba(Xt))

    if Yt is not None:
        printAccuracy(YPredict, Yt, title)
    else:
        return YPredict, YProb


def linearRegression(X, Y, Xt, Yt):

    title = "Linear Regression"

    lr = LinearRegression()
    lr.fit(X, Y)
    YPredict = lr.predict(Xt)


    if Yt is not None:
        printAccuracy(YPredict, Yt, title)
    else:
        return YPredict

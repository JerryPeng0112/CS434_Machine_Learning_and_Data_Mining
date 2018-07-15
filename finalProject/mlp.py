from sklearn.neural_network import MLPClassifier

from util import printAccuracy

def mlp(X, Y, Xt, Yt):

    title = "Multi-layer Perceptron"

    classifier = MLPClassifier(hidden_layer_sizes=(100))
    classifier.fit(X, Y)
    YPredict = classifier.predict(Xt)

    printAccuracy(YPredict, Yt, title)

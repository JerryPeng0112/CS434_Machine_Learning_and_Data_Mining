import numpy as np

"""
Linear Regression Training
"""

def main():
    # Linear regression with dummy variable
    print "====== With Dummy Variable ======\n"
    w = train(True)
    calcASE(w, True)
    # Linear regression without dummy variable
    print "====== Without Dummy Variable ======\n"
    w = train(False)
    calcASE(w, False)

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
    return w

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

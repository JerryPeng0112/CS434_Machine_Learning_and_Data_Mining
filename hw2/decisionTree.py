import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

"""
Decision Trees
"""

class Node:
    def __init__(self, entropy):
        self.type = 1 # type 1: node, type 2: leaf node
        self.entropy = entropy
        self.infoGain = 0
        self.left = None
        self.right = None
        self.value = 0
        self.splitIdx = -1
        self.splitVal = 0

def main():

    # Fetch data
    trainData, testData = getData()

    # Set up depth range for decdion tree
    d = range(1, 7)

    trainErrorRate = []
    testErrorRate = []

    # build decision tree and calculate error rate
    for i in d:

        # build decision tree
        root = buildDecisionTree(trainData, i, 5)

        if i == 1:
            print("\n-------> Information gain for decision stump:")
            print root.infoGain
        # Print the tree
        print("\n-------> Decision Tree for d = " + str(i) + ":")
        printTree(root, 0)

        # calculate error rate
        trainErrorRate.append(calcError(trainData, root))
        testErrorRate.append(calcError(testData, root))

    print("\n-------> Training Error Rate: ")
    print trainErrorRate
    print("\n-------> Testing Error Rate: ")
    print testErrorRate

    plotErrors(d, trainErrorRate, testErrorRate)


def calcError(test, root):
    # Calculate the error based on decision tree and testing data
    correct = 0

    for i, row in enumerate(test):
        x = row['x']
        y = row['y']
        predictY = getPrediction(x, root)

        if predictY == y:
            correct += 1

    errorRate = 1 - correct / float(len(test))
    return errorRate


def getPrediction(x, root):
    # Get prediction based on decision tree
    node = root

    while node.type != 2:
        idx = node.splitIdx
        val = node.splitVal

        if x[idx] < val:
            node = node.left
        else:
            node = node.right

    return node.value


def buildDecisionTree(data, depth, minSize):
    #initialize root with entropy
    root = Node(getEntropy(data))
    #start building tree by splitting root node
    splitNode(root, data, depth, minSize)

    return root


def splitNode(node, data, depth, minSize):
    # Create leaf nodes
    # depth reached
    if depth == 0:
        toLeafNode(node, data)
        return

    # The number of data is below minSize
    if len(data) < minSize:
        toLeafNode(node, data)
        return

    # if the current node entropy is 0:
    if getEntropy(data) == 0:
        toLeafNode(node, data)

    # Split the data
    left, right = getBestSplit(node, data)

    # Recursively split tree
    splitNode(node.left, left, depth - 1, minSize)
    splitNode(node.right, right, depth - 1, minSize)


def toLeafNode(node, data):
    # Turn the node into a leaf node
    node.type = 2
    node.left = None
    node.right = None

    # Decide majority value for the node
    count = len([row for row in data if row['y'] == 1])
    invCount = len(data) - count
    if count > invCount:
        node.value = 1
    else:
        node.value = -1


def getBestSplit(node, data):
    # data for the best split
    best = {
        'leftEntropy': 0,
        'rightEntropy': 0,
        'leftData': [],
        'rightData': [],
    }

    for i, row in enumerate(data):
        for j, val in enumerate(row['x']):

            # split and get entropy, information gain
            left, right = testSplit(data, j, val)

            leftEntropy = getEntropy(left)
            rightEntropy = getEntropy(right)

            leftProb = len(left) / float(len(data))
            infoGain = node.entropy - leftProb * leftEntropy - (1 - leftProb) * rightEntropy

            # if information gain is better, record it
            if (infoGain > node.infoGain):
                best['leftEntropy'] = leftEntropy
                best['rightEntropy'] = rightEntropy
                best['leftData'] = left
                best['rightData'] = right
                node.splitIdx = j
                node.splitVal = val
                node.infoGain = infoGain

    # Create nodes for left and right
    node.left = Node(best['leftEntropy'])
    node.right = Node(best['rightEntropy'])
    return best['leftData'], best['rightData']


def testSplit(data, idx, val):
    # test split by classifying data with a value
    left = []
    right = []

    for i, row in enumerate(data):
        if row['x'][idx] < val:
            left.append(row)
        else:
            right.append(row)

    return left, right


def getEntropy(data):
    entropy = 0
    size = len(data)
    count = len([row for row in data if row['y'] == 1])

    if count != size and count != 0:
        prob = count / float(size)
        entropy = - prob * math.log(prob, 2) - (1 - prob) * math.log((1 - prob), 2)

    return entropy


def printTree(node, depth):
    if node.type == 2:
        prettyPrintLeaf(node, depth)
        return

    prettyPrintNode(node, depth, True)
    printTree(node.left, depth + 1)

    prettyPrintNode(node, depth, False)
    printTree(node.right, depth + 1)

def prettyPrintNode(node, depth, smaller):
    msg = ""
    for i in range(depth):
        msg += "--"
    msg = msg + "Feature " + str(node.splitIdx + 1) + " "
    compare = "<" if smaller else ">="
    msg = msg + compare + " " + str(round(node.splitVal, 2)) + ": "
    msg = msg + "(Entropy: " + str(round(node.entropy, 2))
    msg = msg + ", Gain: " + str(round(node.infoGain, 2)) + ")"
    print msg


def prettyPrintLeaf(node, depth):
    msg = ""
    for i in range(depth):
        msg += "--"
    msg = msg + "Predict Y = " + str(node.value) + " (Entropy: " + str(round(node.entropy, 2)) + ")"
    print msg


def plotErrors(d, train, test):
    # Plot error rates
    plt.plot(d, train, marker='o', label="training error rate")
    plt.plot(d, test, marker='o', label="testing error rate")
    plt.xlabel("# of d")
    plt.ylabel("Error rate")
    plt.legend()
    plt.savefig("DTErrorRate.png")


def getData():
    # get data extract data into varables
    trainData = loadFile("./data/knn_train.csv")
    testData = loadFile("./data/knn_test.csv")

    trainData = [{'x': row[1:], 'y': row[0]} for row in trainData]
    testData = [{'x': row[1:], 'y': row[0]} for row in testData]

    return trainData, testData


def loadFile(path):
    # load data from file and return data matrix
    file = open(path, "r")
    data = np.genfromtxt(file, delimiter=",")
    return data


if __name__ == "__main__":
	main()

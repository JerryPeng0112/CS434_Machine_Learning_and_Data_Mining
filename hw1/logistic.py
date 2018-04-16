import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

"""
Logistic Regression Training
"""

LEARNING_RATE = 0.0001
LEARNING_RATE_REGULAR = 0.000005

def main():
	logisticTraining()
	regularizedTraining()

def logisticTraining():
	inputData = loadData("./data/usps-4-9-train.csv")

	# load data into matrix X and Y
	X = inputData[:, 0:-1] / 255
	Y = inputData[:, -1]

	trainAccuracy, testAccuracy = batchLearn(X, Y)
	print "Training accuracies throughout iterations: "
	print trainAccuracy
	print "Testing accuracies throughout iterations: "
	print testAccuracy
	plotErrors(trainAccuracy, testAccuracy)

def regularizedTraining():
	inputData = loadData("./data/usps-4-9-train.csv")

	# load data into matrix X and Y
	X = inputData[:, 0:-1] / 255
	Y = inputData[:, -1]

	lamb, trainAccuracy, testAccuracy = regularizedBatchLearn(X, Y)
	print "lambda values: "
	print lamb
	print "Training data accuracies: "
	print trainAccuracy
	print "Testing data accuracies: "
	print testAccuracy

def batchLearn(X, Y):
	# get number of independent variables
	print "\n====== Batch Learning ======\n"
	n = X.shape[0]
	features = X.shape[1]
	w = np.zeros(features)
	trainAccuracy = []
	testAccuracy = []

	cond = 100
	while(cond != 0):
		gradient = np.zeros(features)
		for i in range(n):
			result = 1 / (1 + np.exp(-1 * np.dot(np.transpose(w), X[i])))
			gradient += np.dot(result - Y[i], X[i])
		w -= LEARNING_RATE * gradient
		cond -= 1
		trainAccuracy.append(calcError("./data/usps-4-9-train.csv", w))
		testAccuracy.append(calcError("./data/usps-4-9-test.csv", w))

	return trainAccuracy, testAccuracy

def regularizedBatchLearn(X, Y):
	# get number of independent variables

	print "\n====== Regularized Batch Learning ======\n"

	lamb = []
	for i in range(-3, 4, 1):
		lamb.append(10 ** i)
	lamb = np.array(lamb)

	trainAccuracy = []
	testAccuracy = []
	iter = 1

	for item in np.nditer(lamb):
		n = X.shape[0]
		features = X.shape[1]
		w = np.zeros(features)

		cond = 100
		while(cond != 0):
			gradient = np.zeros(features)
			for i in range(n):
				result = 1 / (1 + np.exp(-1 * np.dot(np.transpose(w), X[i])))
				gradient += (np.dot(result - Y[i], X[i]) + item * w)
			w -= LEARNING_RATE_REGULAR * gradient
			cond -= 1

		trainAccuracy.append(calcError("./data/usps-4-9-train.csv", w))
		testAccuracy.append(calcError("./data/usps-4-9-test.csv", w))
		print "Regularized Batch Learning: with lambda"
		print item

	return lamb, trainAccuracy, testAccuracy

def calcError(path, w):
	inputData = loadData(path)
	X = inputData[:, 0:-1]
	Y = inputData[:, -1]

	n = X.shape[0]
	result = np.dot(X, w)
	count = 0
	idx = 0

	for item in np.nditer(result):
		approx = 0
		if item >= 0.5:
			approx = 1
		if Y[idx] == approx:
			count += 1
		idx += 1

	accuracy = count / (n * 1.0)
	return accuracy

def plotErrors(trainAccuracy, testAccuracy):
	# plot training error
	plt.plot(trainAccuracy, label="training data")
	plt.xlabel('# of Iterations')
	plt.ylabel('Accuracy')
	#plot testing error
	plt.plot(testAccuracy, label="testing data")
	plt.xlabel('# of Iterations')
	plt.ylabel('Accuracy')
	plt.savefig('trainingAccuracy.png')


def loadData(path):
	# load data from file and return data matrix
	file = open(path, "r")
	data = np.genfromtxt(file, delimiter=",")
	return data


if __name__ == "__main__":
	main()

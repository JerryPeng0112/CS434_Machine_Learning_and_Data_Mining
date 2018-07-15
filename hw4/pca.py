import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Principal Component Analysis
"""

numEigenVectors = 10
vmin = 0
vmax = 255

def main():
    # Load data from data/data.txt
    X = loadData("./data/data.txt")
    print("--> Data loaded")

    # Calculate top eigen vectors - Problem 1
    meanVec, eigenVec = calcEigenVectors(X, numEigenVectors)

    eigenVec *= vmax

    print(eigenVec)

    # Plot images from vectors - Problem 2
    plotImage(meanVec, "MeanVector.png")
    for i in range(numEigenVectors):
        plotImage(eigenVec[i], "EigenVector" + str(i + 1) + ".png")

    # Find images with best fit to each eigen vector - Problem 3
    imgIndex = findImgBestFit(X, eigenVec)
    print("Image Index with best fit to each eigen vector:")
    print(imgIndex)

    # Print best fit images
    for i, imgIdx in enumerate(imgIndex):
        plotImage(X[imgIdx], "BestFitImg" + str(i + 1) + ".png")


def calcEigenVectors(X, num):
    # Calculate mean vector and covariance matrix
    meanVec = np.mean(X, axis=0)
    covariance = np.cov(X.T)

    # Print mean vector and covariance matrix
    print("--> Mean Vector: ")
    print(meanVec.shape)
    print(meanVec)
    print("--> Covariance Matrix: ")
    print(covariance.shape)
    print(covariance)

    # Get top eigen vectors and eigen values
    eigenVal, eigenVec = np.linalg.eig(covariance)
    eigenVal = eigenVal[0: num]
    eigenVec = (eigenVec.T)[0: num]

    # Print eigen results
    print("--> Eigen values: ")
    print(eigenVal.shape)
    print(eigenVal)
    print("--> Eigen Vectors")
    print(eigenVec.shape)
    print(eigenVec)

    return meanVec, eigenVec


def findImgBestFit(X, eigenVec):
    # Find images best fit with each eigen vector
    imgIndex = []
    for i, vec in enumerate(eigenVec):

        values = []
        #iterate through each sample and dot product with eigen vector
        for j, x in enumerate(X):

            values.append(np.dot(vec.astype(np.float64), x))

        maxValueIndex = values.index(max(values))
        imgIndex.append(maxValueIndex)

    return imgIndex

def plotImage(vec, filename):
    vec = vec.reshape(28, 28).astype(np.float64)
    plt.imshow(vec, cmap="gray", interpolation="nearest")
    plt.savefig(filename)
    plt.clf()


def loadData(path):
    # Load data from file and return data matrix
    file = open(path, "r")
    data = np.genfromtxt(file, delimiter=",")
    return data


if __name__ == "__main__":
	main()

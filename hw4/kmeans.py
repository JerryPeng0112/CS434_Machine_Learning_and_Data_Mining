import numpy as np
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Non-hierarchical clustering - K-means algorithm
"""

Kmin = 2
Kmax = 10
repetition = 10

def main():
    # Load data from data/data.txt
    X = loadData("./data/data.txt")
    print("--> Data loaded")

    # Problem 1
    SSElist = clustering(X, 2)
    plotSSElist(SSElist)

    print("--> Problem 1 complete")
    print("--> Result:")
    print(SSElist)

    # Problem 2
    kSSE = []
    kRange = range(Kmin, Kmax + 1)
    for i in kRange:

        # For each k run several repetition to find best result
        SSEvalues = []
        for j in range(repetition):

            SSEvalues.append(clustering(X, i)[-1])

        smallestSSE = min(SSEvalues)
        kSSE.append(smallestSSE)

        print("k = " + str(i) + " complete..")
        print("Best SSE: " + str(smallestSSE))

    plotKSSE(kRange, kSSE)
    print("--> Problem 2 complete")
    print("--> Result:")
    print(kSSE)


def clustering(X, k):
    # K-mean algorithm for clustering
    numReclustered = X.shape[0]
    clusterAssigns = [-1] * X.shape[0]

    # Randomly select camples as cluster centers
    centers = selectCenters(X, k)
    SSElist = []

    # Adjust clusters until convergence
    while numReclustered != 0:
        # Assign each samples to nearest cluster
        numReclustered = assignCluster(X, clusterAssigns, centers)
        # Update center of cluster
        updateCenters(X, k, clusterAssigns, centers)
        # Calculate new SSE and append to list of SSEs
        SSE = calculateSSE(X, clusterAssigns, centers)
        SSElist.append(SSE)

    return SSElist


def selectCenters(X, k):
    # Randomly select samples as cluster centers
    max = X.shape[0] - 1
    centers = []

    # Generate random index until list has k unique values
    while (len(centers) != k):
        # Generate random value as index of sample
        val = random.randint(0, max)
        # If the randome value not in list, append it to list
        if val not in centers:
            centers.append(val)

    # Assign samples as cluster centers
    for i in range(k):
        centers[i] = X[centers[i]]

    return centers


def assignCluster(X, clusterAssigns, centers):
    # For each sample in the dataset, assign to nearest cluster
    numReclustered = 0
    for i, x in enumerate(X):
        distances = []

        for j, center in enumerate(centers):
            distances.append(getDistance(x, center))

        # Get the index of minimum distance
        minDistanceIndex = distances.index(min(distances))

        # If different cluster assigned, increase by 1
        if clusterAssigns[i] != minDistanceIndex:
            clusterAssigns[i] = minDistanceIndex
            numReclustered += 1

    return numReclustered


def updateCenters(X, k, clusterAssigns, centers):
    # Update the center of each cluster
    for i, center in enumerate(centers):
        samples = []
        # Iterate through samples
        for j, x in enumerate(X):
            # If the sample belongs to the cluster, add it to totalWeight
            if clusterAssigns[j] == i:
                samples.append(x)

        # calculate cluster center
        centers[i] = np.mean(np.array(samples), axis=0)


def calculateSSE(X, clusterAssigns, centers):
    # Calculate the SSE for all clusters
    totalSSE = 0
    for i, center in enumerate(centers):
        # Iterate through samples
        for j, x in enumerate(X):
            # If the sample belongs to cluster, add the SSE
            if clusterAssigns[j] == i:
                totalSSE += getDistance(x, center)

    return totalSSE


def getDistance(a, b):
    return np.linalg.norm(a - b)


def plotSSElist(SSElist):
    # Plot function for problem 1
    # Plot SSE for each iteration when k = 2
    listRange = [ (x + 1) for x in range(len(SSElist))]
    plt.plot(listRange, SSElist, label="k = 2")
    plt.xlabel("# of iterations")
    plt.ylabel("SSE")
    plt.legend()
    plt.savefig("SSElist.png")
    plt.clf()


def plotKSSE(kRange, kSSE):
    plt.plot(kRange, kSSE)
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.legend()
    plt.savefig("SSE_for_each_k.png")
    plt.clf()


def loadData(path):
    # Load data from file and return data matrix
    file = open(path, "r")
    data = np.genfromtxt(file, delimiter=",")
    return data


if __name__ == "__main__":
    main()

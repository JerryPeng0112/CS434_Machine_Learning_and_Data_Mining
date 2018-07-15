from sklearn.cluster import KMeans

from util import printAccuracy

def kmean(X, Y, Xt, Yt):

    title = "K-mean Clustering"

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    YPredict = kmeans.predict(Xt)

    printAccuracy(YPredict, Yt, title)

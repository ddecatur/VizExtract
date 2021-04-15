from PIL import Image
from sklearn.cluster import MiniBatchKMeans as KMeans
import numpy as np
import warnings
from kneed import KneeLocator


def clusterCounts(clt):
    """A function to return the clusters

    clustering code adapted from https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
    """

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def elbowM(arr, kneedleBasic=False):
    """A function that uses the modified version of the
        elbow method to determine an appropriate choice for k

    Parameters:
    -----------
    arr: an image array
    kneedleBasic: bool, a tag to specify whether or not to only use the
        basic kneedle algorithm without our modified algorithm on top

    Returns:
    --------
    kval: int, the appropriate value of k to use for k-means
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = arr.reshape((arr.shape[0] * arr.shape[1], 3))
        maxk = 8
        y = []
        for i in range(1,maxk+1):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(arr)
            y.append(kmeans.inertia_)
        x = range(1, len(y)+1)
        kn = KneeLocator(x, y, S=3.0, online=True, curve='convex', direction='decreasing')
        
        # for basic kneedle usage
        if kneedleBasic:
            return kn.knee
        
        if kn.y_normalized[kn.knee-1] < 0.985:#used to be 99
            return kn.knee+1
        else:
            return kn.knee


def KMeansCluster (imageArr):
    """A function to perform the k-means clustering

    This function takes in an image array and returns a clustering object
    """
    im = Image.fromarray(imageArr)
    im.save('k_placeholder.png')
    k = elbowM(imageArr)
    print("k is: " + str(k))
    clt = KMeans(n_clusters = k)
    imageArr = imageArr.reshape((imageArr.shape[0] * imageArr.shape[1], 3))
    clt.fit(imageArr)
    return clt

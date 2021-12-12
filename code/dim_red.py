import numpy as np
from scipy.spatial import Delaunay
from math import *
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

#This method uses LLE to perform Dimensionality Reduction
#IN: Datapoints, colors for visualization, Boolean if plot is required
#OUT: Plot, returns Embedding
def LLE(Data, colors, plot=False):

    print("Computing LLE embedding")
    X_r, err = manifold.locally_linear_embedding(Data, n_neighbors=12, n_components=2)

    if plot:
        plot_dr(Data, colors)

    return X_r


#This method performs axis Dimensionality Reduction
#In: Datapoints to be embedded, corresponding colors
#Out: Two-Dimensional Embedding
def axis_dr(Data, colors, plot=False):

    print(Data.shape)
    cyl_data = np.zeros_like(Data[:,0:2])
    for idx, point in enumerate(Data):
        cyl_data[idx] = point[2]*point[:2]
    if plot:
        plot_dr(Data, cyl_data, colors)
    print(cyl_data.shape)

    return cyl_data

#Plot the DR
def plot_dr(Data3, Data2, colors):

    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    ax.scatter(Data3[:, 0], Data3[:, 1], Data3[:, 2], c=colors, cmap=plt.cm.Spectral, s=5)
    ax.set_title("Original data")
    ax = fig.add_subplot(212)
    ax.scatter(Data2[:, 0], Data2[:, 1], c=colors, cmap=plt.cm.Spectral, s=5)
    plt.axis("tight")
    plt.xticks([]), plt.yticks([])
    plt.title("Projected data")
    plt.show()

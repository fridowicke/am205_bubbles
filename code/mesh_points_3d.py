import numpy as np
from scipy.spatial import Delaunay
from math import *
import matplotlib.pyplot as plt
#import open3d as o3d
import make_shapes
from sklearn import manifold, datasets
from scipy.spatial import Delaunay

cyl=(make_shapes.make_cylinder(1,1, nh=20, nv=20))

X, color = datasets.make_swiss_roll(n_samples=1500)

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
print("Done. Reconstruction error: %g" % err)
print(X_r.shape)

# ----------------------------------------------------------------------
# Plot result

fig = plt.figure()

ax = fig.add_subplot(211, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, s=5)

ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral, s=5)
plt.axis("tight")
plt.xticks([]), plt.yticks([])
plt.title("Projected data")
plt.show()


tri_rd = Delaunay(X_r)

#Visualize
plt.triplot(X_r[:,0], X_r[:,1], tri_rd.simplices)
plt.show()

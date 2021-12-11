import numpy as np
from scipy.spatial import Delaunay
from math import *
import matplotlib.pyplot as plt
#import open3d as o3d
import make_shapes
from sklearn import manifold, datasets
from scipy.spatial import Delaunay
import dim_red

cyl=(make_shapes.make_cylinder(1,1, nh=20, nv=20))
X, color = datasets.make_swiss_roll(n_samples=1500)


X_r=dim_red.axis_dr(cyl, cyl[:,-1])
tri_cil = Delaunay(X_r)
#plt.triplot(X_r[:,0], X_r[:,1], tri_cil.simplices)
#plt.show()
make_shapes.vis_triang_3d(tri_cil.simplices, cyl)

#Visualize
#plt.triplot(X_r[:,0], X_r[:,1], tri_rd.simplices)
#plt.show()

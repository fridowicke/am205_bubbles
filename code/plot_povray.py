import make_shapes
import numpy as np
from math import *
import dim_red
from scipy.spatial import Delaunay

#(nh*nv)x3 Matrix with the datapoints
cyl=(make_shapes.make_cylinder(1,1, nh=30, nv=30))

#Two-Dimensional Embedding (you do not need this)
X_r=dim_red.axis_dr(cyl, cyl[:,-1], plot=True)

#Triangularization
tri_cil = Delaunay(X_r)
#Tri_cil.simplices: nx3 matrix containing three indices for
#three points describing a triangle
print(tri_cil.simplices)

#Visualization
make_shapes.vis_triang_3d(tri_cil.simplices, cyl)

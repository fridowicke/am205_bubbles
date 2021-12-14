import make_shapes
import numpy as np
from math import *
import dim_red
from scipy.spatial import Delaunay

#(nh*nv)x3 Matrix with the datapoints
cyl, simplices =(make_shapes.make_cylinder(1,1, nh=30, nv=30))

#Visualization
make_shapes.vis_triang_3d(simplices, cyl)
print(cyl.shape)
print(simplices)

import mesh_points_3d
import make_shapes
import numpy as np
from math import *
from scipy.spatial import Delaunay

cyl=(make_shapes.make_cylinder(1,1, nh=30, nv=30))
X_r=dim_red.axis_dr(cyl, cyl[:,-1], plot=True)
tri_cil = Delaunay(X_r)
make_shapes.vis_triang_3d(tri_cil.simplices, cyl)

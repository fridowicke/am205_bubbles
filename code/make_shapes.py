import numpy as np
from math import *

def make_cylinder(h,r,nh=100,nv=100):

    X1 = np.linspace(0, 2*pi,nh)
    X2 = np.linspace(0,h,nv)
    circles = np.array([[(r*cos(x1), r*sin(x1), x2) for x1 in X1]for x2 in X2])
    return np.vstack(circles)

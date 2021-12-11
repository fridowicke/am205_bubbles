import numpy as np
from math import *
import matplotlib.pyplot as plt

def make_cylinder(h,r,nh=100,nv=100):

    X1 = np.linspace(0, 2*pi,nh)
    X2 = np.linspace(0,h,nv)
    circles = np.array([[(r*cos(x1), r*sin(x1), x2) for x1 in X1]for x2 in X2])
    return np.vstack(circles)

def vis_triang_3d(simplices, points):

    #Getting the points
    vertices=[]
    for idc in simplices:
        vertices.append(points[[idc]])

    #Create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    #Plot vertices
    for v in vertices:
        ax.plot(v[:,0],v[:,1],v[:,2], color='orange', linewidth=1)

    plt.show()

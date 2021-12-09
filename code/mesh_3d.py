import numpy as np
from scipy.spatial import Delaunay
from math import *
import matplotlib.pyplot as plt

#Use Delaunay to tesselate a Cylinder
X1 = np.linspace(0, 2*pi,10)
X2 = np.linspace(0,1,10)
circles = np.array([[(cos(x1), sin(x1), x2) for x1 in X1]for x2 in X2])
circle = np.vstack(circles)
#circle=np.array([[0,0,0,],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,0],[1,1,0],[1,0,0]])
tri_cil = Delaunay(circle)




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
        ax.plot(v[:,0],v[:,1],v[:,2], color='grey')

    plt.show()

vis_triang_3d(tri_cil.simplices, circle)

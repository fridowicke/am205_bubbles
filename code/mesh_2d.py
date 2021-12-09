import numpy as np
from scipy.spatial import Delaunay
from math import *
import matplotlib.pyplot as plt

#Use Delaunay to tesselate a circle
X = np.linspace(0, 2*pi,100)
circle1 = np.array([(cos(x), sin(x)) for x in X])
circle2 = np.array([(cos(x), sin(x)) for x in X])/2
circle4 = np.array([(cos(x), sin(x)) for x in X])/4
circle = np.vstack([circle1, circle2, circle4])
print(circle.shape)
tri_cir = Delaunay(circle)

#Visualize
plt.plot(circle1[:,0],circle1[:,1])
plt.plot(circle2[:,0],circle2[:,1])
plt.plot(circle4[:,0],circle4[:,1])
plt.triplot(circle[:,0], circle[:,1], tri_cir.simplices)
plt.show()


#Use Delaunay to tesselate a cloud of random points
xx = np.random.rand(100)
yy = np.random.rand(100)
xy = np.array([xx,yy]).T
tri_rd = Delaunay(xy)

#Visualize
plt.triplot(xx, yy, tri_rd.simplices)
plt.show()

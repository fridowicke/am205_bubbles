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

#Takes input with 3 points and calculates the angle
def angle(X):
    a,b,c = X
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def Mean_curve(X):
    Gradient_A = 0
    p = X
    for i in range() #number of adjacent triangles
    #for each adjacent tri PQR with angle alpha (pqr) an beta prq
    alpha = angle(p,q,r)
    beta = angle(p,r,q)
    Gradient_A += 1/2*(np.cot(alpha)*(p-q)+np.cot(beta)*(p-r))

#####Need an array with the point together with an array with all adjacent points#####
# @Leon: For the point i, all nonzero indices of row i are connected to point i
# You can get them by computing np.argwhere(Adj_mat[i]!=0)

# Create Adjacency Matrix for this
n = cyl.shape[0]                     # Number of points
Adj_mat = np.zeros((n,n))            #Adjacency Matrix (j,i)=(i,j)=1 if there is a connection between Nodes i and j
for simplex in tri_cil.simplices:    #create the ones:
    x,y,z = simplex
    Adj_mat[x,y] = Adj_mat[y,x] = 1  #There is a vertex between x,y => Mat[x,y]=Mat[y,x]=1
    Adj_mat[x,z] = Adj_mat[z,x] = 1  #There is a vertex between x,z => Mat[x,z]=Mat[z,x]=1
    Adj_mat[y,z] = Adj_mat[z,y] = 1  #There is a vertex between y,z => Mat[y,z]=Mat[z,y]=1

import make_shapes
import numpy as np
from math import *
import dim_red
from scipy.spatial import Delaunay
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la

#(nh*nv)x3 Matrix with the datapoints
cyl=(make_shapes.make_cylinder(1,1, nh=30, nv=30))
boundaries1 = np.argwhere(cyl[:,2]==0).flatten()
boundaries2 = np.argwhere(cyl[:,2]==1).flatten()
boundaries = np.array((boundaries1, boundaries2)).flatten()



#Two-Dimensional Embedding (you do not need this)
X_r=dim_red.axis_dr(cyl, cyl[:,-1], plot=True)
#Triangularization
tri_cil = Delaunay(X_r)
#print(tri_cil.simplices)
#Visualization
#make_shapes.vis_triang_3d(tri_cil.simplices, cyl)

#Takes input with 3 points and calculates the angle
def angle(X):
    a,b,c = X
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

#Get the Triangles that a given vertex is connected to
def get_triangles(vertices):
    triangles = [[] for idx in range(cyl.shape[0])]
    for x,y,z in vertices:
        triangles[x].append([y,z])
        triangles[y].append([x,z])
        triangles[z].append([x,y])
    return triangles

#Compute the gradients
def compute_gradients(points, triangles):
    def Mean_curve(point):
        Gradient_A = np.zeros(3)
        p = points[point]
        for q,r in triangles[point]: #number of adjacent triangles
            q, r = points[q], points[r]
            alpha, beta = angle([p,q,r]), angle([p,r,q])
            Gradient_A += 1/2*(cot(alpha)*(p-q)+cot(beta)*(p-r))
        return Gradient_A

    gradients = np.zeros_like(points)
    for idx in range(points.shape[0]):
        gradients[idx] = Mean_curve(idx)
        #print(Mean_curve(idx))
    return gradients

#Optimize the surface
def opt_surface(points, triangles, boundary_points, epsilon = 0.01):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[boundary_points][:,0],points[boundary_points][:,1],points[boundary_points][:,2])
    plt.show()

    for idx in range(100):
        #Plotting
        if idx%10==1:
            #make_shapes.vis_triang_3d(tri_cil.simplices, points)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points[:,0],points[:,1],points[:,2])
            plt.show()
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.quiver(points[:,0],points[:,1],points[:,2],grads[:,0],grads[:,1],grads[:,2])
            plt.show()

        #Computing the Gradients
        grads=compute_gradients(points, triangles)
        grads= grads*(-10/np.max(np.abs(grads)))*epsilon
        #Reducing the size of the gradient and inverting

        #Removing too large gradients (not sure if that makes sense)
        norms = np.array([la.norm(grad) for grad in grads])
        grads[np.argwhere(norms>norms.mean())]=0

        #Enforcing the boundary conditions
        grads[boundary_points]=0

        #Updating the points
        points=points-grads
    return points

#Get the simplices, where edges above a tolerance (too long) are removed
def get_simplices(points, s_init, tol=0.3):
    s_final = []
    for simplex in s_init:
        x,y,z = points[simplex]
        dist = np.array([la.norm(x-y),la.norm(y-z),la.norm(x-z)])
        print(np.max(dist))
        if np.max(dist)<tol:
            print(np.max(dist))
            s_final.append(simplex)
    return np.array(s_final)

simplices = get_simplices(cyl, tri_cil.simplices)
triangles =get_triangles(simplices)
gradients = compute_gradients(cyl, triangles)


opt = opt_surface(cyl, triangles, boundaries)
#make_shapes.vis_triang_3d(simplices, cyl)

import make_shapes
import numpy as np
from math import *
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la

#(nh*nv)x3 Matrix with the datapoints
cyl, tri_cil = (make_shapes.make_cylinder(0.5,1, nh=60, nv=50, plot=True))
boundaries1 = np.argwhere(cyl[:,2]==0).flatten()
boundaries2 = np.argwhere(cyl[:,2]==0.5).flatten()
boundaries = np.hstack((boundaries1, boundaries2)).flatten()
print("boundaries",boundaries,"boundaries")

#fo, tri_fo=make_shapes.make_fo()


#Two-Dimensional Embedding (you do not need this)

#Triangularization
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
def get_triangles(vertices, points):
    triangles = [[] for idx in range(points.shape[0])]
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
def opt_surface(points, triangles, boundary_points, epsilon = 0.01, plot_boundaries=True):

    #Plot the Boundary conditions
    if plot_boundaries:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.suptitle("Boundary points")
        ax.scatter(points[boundary_points][:,0],points[boundary_points][:,1],points[boundary_points][:,2])
        plt.show()

    for idx in range(20):
        print(idx)
        #Computing the Gradients
        grads=compute_gradients(points, triangles)

        #Removing too large gradients (not sure if that makes sense)
        norms = np.array([la.norm(grad) for grad in grads])
        grads[np.argwhere(norms>norms.mean())]=0

        #Reducing the size of the gradient and inverting
        grads= -epsilon*grads*(10/np.max(np.abs(grads)))
        for idx1 in range(grads.shape[0]):
            #print(la.norm(grads[idx])*np.array([-points[idx,0],-points[idx,1],0]))
            grads[idx1] = la.norm(grads[idx1])*np.array([-points[idx1,0],-points[idx1,1],0])
        #Enforcing the boundary conditions
        grads[boundary_points]=0

        #Plotting
        if idx%10==0:
            make_shapes.vis_triang_3d(tri_cil, points)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points[:,0],points[:,1],points[:,2])
            plt.show()
            #fig = plt.figure()
            #ax = fig.gca(projection='3d')
            #ax.quiver(points[:,0],points[:,1],points[:,2],30*grads[:,0],30*grads[:,1],30*grads[:,2])
            #plt.show()

        #Updating the points
        points=points+grads

    return points

#Get the simplices, where edges above a tolerance (too long) are removed
def get_simplices(points, s_init, tol=3):
    s_final = []
    for simplex in s_init:
        x,y,z = points[simplex]
        dist = np.array([la.norm(x-y),la.norm(y-z),la.norm(x-z)])
        #print(np.max(dist))
        if np.max(dist)<tol:
            #print(np.max(dist))
            s_final.append(simplex)
    return np.array(s_final)

simplices = get_simplices(cyl, tri_cil)
triangles =get_triangles(simplices, cyl)
gradients = compute_gradients(cyl, triangles)
#simplices = get_simplices(fo, tri_fo)
#print(simplices.shape,fo.shape)
#triangles =get_triangles(simplices, fo<)
#gradients = compute_gradients(fo, triangles)

opt = opt_surface(cyl, triangles, boundaries)
#opt = opt_surface(fo, triangles, boundaries)
#make_shapes.vis_triang_3d(simplices, cyl)

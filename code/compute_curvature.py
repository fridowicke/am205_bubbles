import make_shapes
import numpy as np
from math import *
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize


#Takes input with 3 points and calculates the angle
def angle(X):
    a,b,c = X
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

#Get the Triangles that a given vertex is connected to
def get_triangles(points, vertices):
    triangles = [[] for idx in range(points.shape[0])]
    for x,y,z in vertices:
        if x==min([x,y,z]):
            triangles[x].append([y,z])
        elif y==min([x,y,z]):
            triangles[y].append([x,z])
        else:
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
            if tan(alpha)!=0 and tan(beta)!=0:
                Gradient_A += 1/2*(cot(alpha)*(p-q)+cot(beta)*(p-r))
            else:
                print("x")
        return Gradient_A

    gradients = np.zeros_like(points)
    for idx in range(points.shape[0]):
        gradients[idx] = Mean_curve(idx)
        #print(Mean_curve(idx))
    return gradients

#Optimize the surface
def opt_surface(points, triangles, boundary_points, epsilon = 0.01, plot_boundaries=False):

    #Plot the Boundary conditions
    if plot_boundaries:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.suptitle("Boundary points")
        ax.scatter(points[boundary_points][:,0],points[boundary_points][:,1],points[boundary_points][:,2])
        plt.show()

    for idx in range(30):
        print(idx)
        #Computing the Gradients
        grads=compute_gradients(points, triangles)
        #Removing too large gradients (not sure if that makes sense)
        #norms = np.array([la.norm(grad) for grad in grads])
        #grads[np.argwhere(norms>norms.mean())]=0

        #Reducing the size of the gradient and inverting
        grads= -epsilon*grads*(10/np.max(np.abs(grads)))
        print(grads)
        #for idx1 in range(grads.shape[0]):
            #print(la.norm(grads[idx])*np.array([-points[idx,0],-points[idx,1],0]))
        #    grads[idx1] = la.norm(grads[idx1])*np.array([-points[idx1,0],-points[idx1,1],0])
        #Enforcing the boundary conditions
        grads[boundary_points]=0

        #Plotting
        if idx%100==200:
            #make_shapes.vis_triang_3d(tri_cil, points)
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

def get_area(points, triangles):
    area = np.zeros(len(triangles))
    area=[]
    for idx, triangle in enumerate(triangles):
        for id in range(0):#range(len(triangle)):
            t = [idx, triangle[id][0], triangle[id][1]]
            x,y,z = points[t]
            x1,x2,x3 = x
            y1,y2,y3 = y
            z1,z2,z3 = z
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            v=np.array(points[t])
            ax.plot(v[0:2,0],v[0:2,1],v[0:2,2], color='orange', linewidth=1)
            ax.plot(v[1:3,0],v[1:3,1],v[1:3,2], color='green', linewidth=1)
            ax.plot(v[[0,2],0],v[[0,2],1],v[[0,2],2], color='blue', linewidth=1)

            fig.suptitle(f"The area is {np.abs(x1*y2+x2*y3+x3*y1-y1*x2-y2*x3-y3*x1)/2}n\n,{t}")
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(0, 1)
            plt.show()
    for idx, triangle in enumerate(triangles):
        for id in range(len(triangle)):
            t = [idx, triangle[id][0], triangle[id][1]]
            x,y,z = points[t]
            x1,x2,x3 = x
            y1,y2,y3 = y
            z1,z2,z3 = z
            p1 = y-x
            p2 = z-x
            u = np.cross(p1,p2)
            area.append(la.norm(u)/2)
    print(np.sum(np.array(area)))
    return np.sum(np.array(area))

def get_area_boundaries(points, triangles, boundaries, boundarie_values):
    points=points.reshape([-1,3])
    points[boundaries] = boundarie_values
    return get_area(points, triangles)



cyl, tri_cil = (make_shapes.make_cylinder(1,1))
boundaries1 = np.argwhere(cyl[:,2]==0).flatten()
boundaries2 = np.argwhere(cyl[:,2]==1).flatten()
boundaries = np.hstack((boundaries1, boundaries2)).flatten()
boundarie_values=cyl[boundaries]
triangles =get_triangles(cyl, tri_cil)
triangles_opt=minimize(get_area_boundaries, cyl, args=(triangles, boundaries, boundarie_values))


#compute_gradients(cyl, triangles)
#print(get_area(cyl, triangles),"SURFACE AREA")
#gradients = compute_gradients(cyl, tri_cil)
#make_shapes.vis_triang_3d(tri_cil, cyl)
#opt = opt_surface(cyl, triangles, boundaries)

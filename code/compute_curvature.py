#
#
#
import make_shapes
import numpy as np
from math import *
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize
from render_output import render_output
from simple_catenoid import get_curve

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
    tri, tri_sg = [[] for idx in range(points.shape[0])], [[] for idx in range(points.shape[0])]
    for x,y,z in vertices:
        if x==min([x,y,z]):
            tri_sg[x].append([y,z])
        elif y==min([x,y,z]):
            tri_sg[y].append([x,z])
        else:
            tri_sg[z].append([x,y])
        tri[x].append([y,z])
        tri[y].append([x,z])
        tri[z].append([x,y])
    return tri, tri_sg

def get_grad(points, triangles, boundaries, boundarie_values):
    points.reshape([3,-1])
    grad = compute_gradients(points, triangles)
    grad[boundaries] = 0
    return grad.flatten()

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
        #print(gradients[idx],Mean_curve(idx))
        gradients[idx] = Mean_curve(idx)
        #print(Mean_curve(idx))
    return gradients

#Optimize the surface
def opt_surface(points, triangles, boundary_points, tol=1e-06, plot_opt=True, plot_gradient=True, n_iter=10000):

    #Convert the triangles to the Format used in Rendering
    simplices = convert_tri_simplices(triangles)
    areas=[]
    for idx in range(n_iter):

        print(idx)

        #Create the PovRay File
        render_output(points, simplices, idx)

        #Keeping track of the Area
        areas.append(get_area(points, triangles))

        #Computing the Gradients
        grads=compute_gradients(points, triangles)

        #Enforcing Boundary conditions
        grads[boundary_points]=0

        #implementing line search
        eta = minimize(grad_area, 0, args=(grads, points, triangles)).x[0]

        #Scaling the Gradients
        grads= eta*grads

        #Plotting
        if idx%1==0:

            ### PLOTTING THE SHAPE
            vertices=[]
            for idc in simplices:
                vertices.append(points[[idc]])

            #Create figure
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')

            #Plot vertices
            for v in vertices:
                ax.plot(v[0:2,0],v[0:2,1],v[0:2,2], color='orange', linewidth=1)
                ax.plot(v[1:3,0],v[1:3,1],v[1:3,2], color='green', linewidth=1)
                ax.plot(v[[0,2],0],v[[0,2],1],v[[0,2],2], color='blue', linewidth=1)

            fig.suptitle(f"Shape after{idx} Iterations")
            plt.savefig(f"./plots/Shape after{idx} Iterations", dpi=1000)


            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.scatter(points[:,0],points[:,1], points[:,2])
            fig.suptitle(f"Pointcloud and Optimal Solution after {idx} iterations")

            if plot_opt:
                X1 = np.linspace(0, 2*pi*((nh-1)/(nh)),nh)
                X2 = np.linspace(0,h,nv)
                R = get_curve([0,r],[h,r],X=np.linspace(0,h,nv))
                opt=[]
                for x1 in X1:
                    for idx1 in range(nv):
                        opt.append((R[idx1]*cos(x1), R[idx1]*sin(x1), X2[idx1]))
                opt=np.array(opt)
                ax.scatter(opt[:,0],opt[:,1],opt[:,2], color='green')
            plt.savefig(f"./plots/Pointcloud and Optimal Solution after {idx} iterations", dpi=1000)

        if idx==0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.quiver(points[:,0],points[:,1],points[:,2],grads[:,0],grads[:,1],grads[:,2])
            plt.suptitle(f"Initial Gradients")
            plt.savefig("./plots/Gradients.png", dpi=1000)

        if np.abs(eta)<tol:
            return points, np.array(areas), idx




        #Updating the points
        points=points+grads

    return points, np.array(areas), idx

def grad_area(epsilon, gradient, points, triangles):
    return get_area(points+epsilon*gradient, triangles)

def get_area(points, triangles):
    area=[]
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
    return np.sum(np.array(area)/3)

def get_area_boundaries(points, triangles, boundaries, boundarie_values):
    points=points.reshape([-1,3])
    points[boundaries] = boundarie_values
    return get_area(points, triangles)



#convert to simplices format for Pov-Raying.
def convert_tri_simplices(triangles):
    simplices=[]
    for idx1, tr in enumerate(triangles):
        for t in tr:
            simplices.append([idx1,t[0],t[1]])
    return simplices


#
h=1
r=1
nh= 24
nv = int((h/r)*nh)
cyl, tri_cil = (make_shapes.make_cylinder(h,r,nh,nv,plot=True))
#cyl,tri_cil = make_shapes.make_tube(nh=nh,nv=nv)
#cyl, tri_cil = make_shapes.make_fo(plot=False)
boundaries1 = np.argwhere(cyl[:,2]==0).flatten()
boundaries2 = np.argwhere(cyl[:,2]==h).flatten()
boundaries = np.hstack((boundaries1, boundaries2)).flatten()
tri, tri_sg =get_triangles(cyl, tri_cil)
points, areas, idx =opt_surface(cyl, tri, boundaries)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(areas.shape[0]), areas)
fig.suptitle("Surface Area after each Iteration")
plt.xlabel("Iteration")
plt.ylabel("Surface Area")
plt.savefig("./plots/Surface over iterations.png", dpi=1000)

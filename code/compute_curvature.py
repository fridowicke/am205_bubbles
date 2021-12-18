import make_shapes
import numpy as np
from math import *
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize
import plot_povray
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
    #Ordering?
    #There should be some sign information here
    def Mean_curve(point):
        Gradient_A = np.zeros(3)
        p = points[point]
        print(point, len(triangles[point]))
        for q,r in triangles[point]: #number of adjacent triangles
            q, r = points[q], points[r]
            alpha, beta = angle([p,q,r]), angle([p,r,q])
            if tan(alpha)!=0 and tan(beta)!=0:
                Gradient_A += 1/2*(cot(alpha)*(p-q)+cot(beta)*(p-r))
            else:
                print("x")
        #print(Gradient_A)
        return Gradient_A

    gradients = np.zeros_like(points)
    for idx in range(points.shape[0]):
        #print(gradients[idx],Mean_curve(idx))
        gradients[idx] = Mean_curve(idx)
        #print(Mean_curve(idx))
    return gradients

#Optimize the surface
def opt_surface(points, triangles, boundary_points, eta =0.0001, plot_boundaries=False, plot_opt=True, plot_gradient=True):

    #Plot the Boundary conditions
    if plot_boundaries:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.suptitle("Boundary points")
        ax.scatter(points[boundary_points][:,0],points[boundary_points][:,1],points[boundary_points][:,2])
        plt.show()
    areas=[]
    gradient_values=[]
    for idx in range(1000000):
        areas.append(get_area(points, triangles))
        print(idx, get_area(points, triangles))
        #Computing the Gradients
        grads=compute_gradients(points, triangles)
        #Reducing the size of the gradient and inverting
        gradient_values.append(grads[::3].std())
        #implement line search
        #grads = -grads*eta*(10/(la.norm(grads.mean(axis=0))))
        eta = minimize(grad_area, 0, args=(grads, points, triangles)).x[0]
        grads= eta*grads
        grads[boundary_points]=0
        gradient_values.append(grads[::3].std())

        #Plotting
        if idx%15==0:
            print(idx,grads)
            make_shapes.vis_tr_3d(points,triangles)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(points[:,0],points[:,1],points[:,2])
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)

            if plot_opt:
                X1 = np.linspace(0, 2*pi*((nh-1)/(nh)),nh)
                X2 = np.linspace(0,h,nv)
                R = get_curve([0,r],[h,r],X=np.linspace(0,h,nv))
                opt=[]
                for x1 in X1:
                    for idx1 in range(nv):
                        print("R",R[idx1])
                        opt.append((R[idx1]*cos(x1), R[idx1]*sin(x1), X2[idx1]))
                opt=np.array(opt)
                ax.scatter(opt[:,0],opt[:,1],opt[:,2], color='green')

            plt.show()

            Y = np.array(areas)
            Z = np.array(gradient_values)
            X = np.arange(len(areas))
            plt.plot(X,Y)
            #plt.plot(X,Z)
            plt.show()
            if plot_gradient:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.quiver(points[:,0],points[:,1],points[:,2],grads[:,0],grads[:,1],grads[:,2])
                plt.show()

        #Updating the points
        points=points+grads

    return points

def grad_area(epsilon, gradient, points, triangles):
    return get_area(points+epsilon*gradient, triangles)

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
    print(np.sum(np.array(area))/3)
    if False:#np.sum(np.array(area))/3<5.9:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[:,0],points[:,1],points[:,2])
        plt.show()
    return np.sum(np.array(area)/3)

def get_area_boundaries(points, triangles, boundaries, boundarie_values):
    points=points.reshape([-1,3])
    points[boundaries] = boundarie_values

    #LEON HERE:
    #plot_povray.function(whatever)
    #points: nx3 of the points
    #triangles -> triangles
    #boundarie_values: boundary points
    #boundaries: indices of the boundary points
    #glob: identifier
    #glob+=1
    return get_area(points, triangles)

#order the triangles
def order_triangle(triangles, boundaries):
    #Empty list of ordered triangles
    ordered_triangles=[]
    #Loop over the triangle "midpoints"
    for idx_tr, tr in enumerate(triangles):

        #Exclude boundary points
        if not(idx_tr in boundaries):

            #initialize the first element
            ordered_tr = [tr[0]]
            prevpoint= tr[0][0]
            nextpoint= tr[0][1]
            print(tr)

            #look for the next tuple and append it
            while len(ordered_tr)<len(tr):
                print(prevpoint, nextpoint)
                idx = 0
                while (not nextpoint in tr[idx]) or (prevpoint in tr[idx]) :
                    idx+=1
                ordered_tr.append(tr[idx])
                idx_next=0
                if tr[idx][0]==nextpoint:
                    idx_next=1
                prevpoint = nextpoint
                nextpoint = tr[idx][idx_next]
            ordered_triangles.append(ordered_tr)

        #append empty list for Boundary points
        else:
            ordered_triangles.append([])
    return ordered_triangles

def opt(points, triangles, boundary_points, eta =0.0001, plot_boundaries=False, plot_opt=True, plot_gradient=True):



    for idx in range(1000):
        print(idx, get_area(points, triangles))
        grads=compute_gradients(points, triangles)
        #Reducing the size of the gradient and inverting
        #implement line search
        print(np.max([la.norm(grad) for grad in grads]))
        grads = -grads*eta*(10/np.max([la.norm(grad) for grad in grads]))
        #eta = minimize(grad_area, 0, args=(grads, points, triangles)).x[0]
        #grads= eta*grads
        grads[boundary_points]=0
        #Updating the points
        points=points+grads

    return points

#glob=0
h=1
r=1
nh= 34
nv = 6
cyl, tri_cil = (make_shapes.make_cylinder(h,r,nh,nv,plot=True))
#cyl,tri_cil = make_shapes.make_tube(nh=nh,nv=nv)
#cyl, tri_cil = make_shapes.make_fo(plot=False)
boundaries1 = np.argwhere(cyl[:,2]==0).flatten()
boundaries2 = np.argwhere(cyl[:,2]==h).flatten()
boundaries = np.hstack((boundaries1, boundaries2)).flatten()
boundarie_values=cyl[boundaries]
tri, tri_sg =get_triangles(cyl, tri_cil)
tri = order_triangle(tri, boundaries)
# LEON: Here the optimization takes place
#triangles_opt=minimize(get_area_boundaries, cyl, args=(tri, boundaries, boundarie_values), jac=get_grad)
#triangles_opt=minimize(get_area_boundaries, cyl, args=(tri, boundaries, boundarie_values))

#cylopt=opt(cyl, tri, boundaries)
cylopt=opt_surface(cyl, tri, boundaries)
#compute_gradients(cyl, triangles)
#print(get_area(cyl, tri),"SURFACE AREA")
#gradients = compute_gradients(cyl, tri_cil)
#make_shapes.vis_triang_3d(tri_cil, cylopt = opt_surface(cyl, tri, boundaries)

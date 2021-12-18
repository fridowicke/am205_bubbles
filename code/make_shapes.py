import numpy as np
import numpy.linalg as la
from math import *
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import dim_red


def get_sphere(plot=True):
    sphere = [x.split(' ') for x in open(f"./Test_sphere-2.obj").readlines()]
    faces    = []
    vertices = []

    for element in sphere:
        #print(element)
        if element[0] =='f':
            vals = [x for x in element[1:4]]
            ap=[]
            for v in vals:
                if v[1]=='/':
                    ap.append(int(float(v[0])))
                elif v[2]=='/':
                    ap.append(int(float(v[0:2])))
                elif v[3]=='/':
                    ap.append(int(float(v[0:3])))
            faces.append(ap)
        if element[0] == 'v':
            vertices.append([float(x)-48 for x in element[1:4]])
    print(faces)
    vertices=np.array(vertices)
    if plot:
        vis_triang_3d(vf)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(vertices[:,0],vertices[:,1],vertices[:,2])
        plt.show()

def make_cylinder(h,r,nh=31,nv=41, plot=False):

    #maybe triangles appear more than once
    X1 = np.linspace(0, 2*pi*((nh-1)/(nh)),nh)
    #X1 = np.linspace(0, 2*pi,nh)
    X2 = np.linspace(0,h,nv)
    circles = np.array([[(r*cos(x1), r*sin(x1), x2) for x1 in X1]for x2 in X2])
    points=np.vstack(circles)
    #X_r=dim_red.axis_dr(points, points[:,-1], plot=plot)
    simplices=[]
    n=nh*nv
    for idx in range(n):
        if idx+nh<n:
            simplices.append([idx, idx+1, idx+nh])
            simplices.append([idx, idx+nh-1, idx+nh])
    simplices=np.array(simplices)
    if plot:
        vis_triang_3d(points, simplices)
    return points, simplices


def make_tube(bottom_shape='circle', bottom_size=1, top_shape='square', top_size=1, nh=32,nv=8,plot=True):
    bottom=make_shape(bottom_shape, bottom_size, nh)
    top=make_shape(top_shape, top_size, nh)
    top[:,2]=1
    points = np.zeros((nh,nv,3))
    for idx1 in range(nh):
        b=bottom[idx1]
        t=top[idx1]
        for idx2 in range(nv):
            points[idx1,idx2]=b+(idx2/nv)*(t-b)
    points_final=[]
    for idx1 in range(nv):
        for idx2 in range(nh):
            points_final.append(list(points[idx2,idx1]))

    points = np.array(points_final)
    print(points.shape)
    simplices=[]
    n = nh*nv
    for idx in range(n):
        if idx+nh<n:
            simplices.append([idx, idx+1, idx+nh])
            simplices.append([idx, idx+nh-1, idx+nh])
    simplices=np.array(simplices)
    simplices=np.array(simplices)
    if plot:
        vis_triang_3d(points, simplices)
    return points, simplices


def make_shape(shape, size, n):
    size*=0.5
    if shape=='circle':
        X1 = np.linspace(0, 2*pi*((n-1)/n),n)
        return np.array([(size*cos(x1), size*sin(x1), 0) for x1 in X1])
    if shape == 'square':
        points=[]
        for idx in range(0,int(n/8)):
            points.append([size,8*idx*size/n,0])
        for idx in range(int(2*n/8)):
            points.append([size-8*idx*size/n, size,0])
        for idx in range(int(2*n/8)):
            points.append([-size, size-8*idx*size/n,0])
        for idx in range(int(2*n/8)):
            points.append([-size+8*idx*size/n, -size,0])
        for idx in range(0,int(n/8)):
            points.append([size,8*idx*size/n-size,0])
        return np.array(points)

    return X[X[:,0]<np.inf]

def make_fo(width=10, height=5,r=[1,1], h=[1,-1], pos=[[2.5,2.5],[7.5,2.5]],plot=True, nh=20, nv=5,n=20):
    r, h, pos = np.array(r), np.array(h), np.array(pos)

    #Determine the range of the cylindrical shapes
    ranges=np.zeros(r.shape[0])
    for idx, point in enumerate(pos):
        min_dist=np.inf
        for point2 in pos:
            dist=la.norm(point-point2)
            if dist>0 and dist<min_dist:
                min_dist=dist
        ranges[idx] = np.min((point[0],point[1], width-point[0], height-point[1], min_dist/2))

    #Compute the z value for a given point
    def get_h(point1):
        for idx, point2 in enumerate(pos):
            if la.norm(point1-point2)<ranges[idx]:
                return ((ranges[idx]-la.norm(point1-point2))/(ranges[idx]-r[idx]))**2*h[idx]
                return ((ranges[idx]-(la.norm(point1-point2))/(ranges[idx]-r[idx])))**2*h[idx]
        return 0

        #point=np.array(point)
        #min_dist = np.min((point[0],point[1], width-point[0], height-point[1]))
        #value = hl*rl*(1/la.norm(point-posl)) + hr*rr*(1/la.norm(point-posr))
        #return min_dist*value
    max = np.max([width,height])
    X1 = np.linspace(0, 2*pi,nh)

    #Create a list for the points and the boundary points
    points=[]
    boundaries=[]
    #Create the points along the cylindrical shapes
    count=0
    for idx, point in enumerate(pos):
        [boundaries.append(count+x) for x in range(nh)]
        R=np.square(np.linspace(sqrt(r[idx]),sqrt(ranges[idx]),nv))
        circles1 = np.array([[(x2*cos(x1)+point[0], x2*sin(x1)+point[1]) for x1 in X1]for x2 in R])
        points.append(np.vstack(circles1))
        count+=points[0].shape[0]
        #print(np.vstack(circles1).shape,"-------------")


    X = []
    #add the points from the grid that arent in one of the circle's radii
    for idx1 in np.linspace(0,width,n):
        for idx2 in np.linspace(0,height,n):
            add=True
            for idx, point in enumerate(pos):
                if la.norm([idx1,idx2]-point)<=ranges[idx]:
                    add=False
            if add:
                X.append([idx1,idx2])
    points.append(np.array(X))
    circle=np.vstack(points)

    #Create the Three-dimensional points
    ret=[]
    for point in circle:
        x = point[0]
        y = point[1]
        ret.append([point[0],point[1], get_h(point)])
    ret=np.array(ret)

    #Create the Meshing
    simplices=Delaunay(ret[:,0:2]).simplices

    #remove the edges of the circles
    dels=[]
    for idx, sim in enumerate(simplices):
        if sim[0] in boundaries or sim[1] in boundaries or sim[2] in boundaries:
            dels.append(idx)
    idc = np.arange(simplices.shape[0])
    idc = np.delete(idc,dels)
    simplices=simplices[idc]
    #Edges removed

    #add back the top row
    add_simplices=[]
    for idx1 in range(0,len(boundaries),nh):
        point0 = boundaries[idx1]
        add_simplices.append([point0, point0+nh-1, point0+nh])
        add_simplices.append([point0, point0+nh-1, point0-1+2*nh])
        for idx2 in range(nh-1):
            point=boundaries[idx1+idx2]
            add_simplices.append([point, point+1, point+1+nh])
            add_simplices.append([point, point+nh, point+1+nh])
    add_simplices = np.array(add_simplices)
    simplices=np.vstack([simplices, add_simplices])

    #Plot the points
    if plot:
        vis_triang_3d(ret, simplices)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(ret[:,0],ret[:,1],ret[:,2])
        plt.show()
    return ret, simplices


def mesh(nh=10,nv=5, width=1, height=0.5, plot=False):
    points=[]
    for idx1 in range(nv):
        for idx2 in range(nh):
            points.append([idx2*width/nh, idx1*height/nv,0])
    points=np.array(points)
    print(points)
    simplices=[]
    n=nh*nv
    for idx in range(n):
        if idx+nh<n and (idx+1)%nh!=0:
            simplices.append([idx, idx+1, idx+nh])
            simplices.append([idx+1, idx+nh, idx+nh+1])
        simplices1=np.array(simplices)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1)
    if plot:
        for s in simplices1:
            v=points[s]
            ax.plot(v[0:2,0],v[0:2,1], color='orange', linewidth=1)
            ax.plot(v[1:3,0],v[1:3,1], color='green', linewidth=1)
            ax.plot(v[[0,2],0],v[[0,2],1], color='blue', linewidth=1)
        plt.show()





def vis_triang_3d(points, simplices):

    #Getting the points
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
        #ax.plot(v[:,-1:-2:-1],v[:,-1:-2:-1],v[:,-1:-2:-1], color='orange', linewidth=1)

    plt.show()

def vis_tr_3d(points, simplices):

    #Getting the points
    vertices=[]
    for idx in range(len(simplices)):
        for idx2 in simplices[idx]:
            vertices.append(points[np.array([idx2[0],idx2[1],idx])])

    #Create figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    #Plot vertices
    for v in vertices:
        ax.plot(v[0:2,0],v[0:2,1],v[0:2,2], color='orange', linewidth=1)
        ax.plot(v[1:3,0],v[1:3,1],v[1:3,2], color='green', linewidth=1)
        ax.plot(v[[0,2],0],v[[0,2],1],v[[0,2],2], color='blue', linewidth=1)
        #ax.plot(v[:,-1:-2:-1],v[:,-1:-2:-1],v[:,-1:-2:-1], color='orange', linewidth=1)

    plt.show()

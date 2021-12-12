import numpy as np
import numpy.linalg as la
from math import *
import matplotlib.pyplot as plt

def make_cylinder(h,r,nh=100,nv=100):

    X1 = np.linspace(0, 2*pi,nh)
    X2 = np.linspace(0,h,nv)
    circles = np.array([[(r*cos(x1), r*sin(x1), x2) for x1 in X1]for x2 in X2])
    return np.vstack(circles)

def make_tube(bottom_shape='circle', bottom_size=1, top_shape='circle', top_size=1, nh=16,nv=16):
    bottom=make_shape(bottom_shape, bottom_size, nh)
    top=make_shape(top_shape, top_size, nh)
    top[:,2]=1
    points = np.zeros((nh*nv,3))
    for idx1 in range(nh):
        b=bottom[idx1]
        t=top[idx1]
        for idx2 in range(nv):
            points[idx1*nh+idx2]=b+(idx2/nv)*(t-b)
    return points


def make_shape(shape, size, n):
    size*=0.5
    if shape=='circle':
        X1 = np.linspace(0, 2*pi,n)
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

def make_fo(width=10, height=5,rl=1,rr=1, hl=1,hr=-1, posl=[2.5,2.5], posr=[7.5,2.5], h=0.2):
    nx = int(width/h)
    ny = int(height/h)
    X = np.inf*np.ones((nx*ny, 3))
    for idx1 in range(nx):
        for idx2 in range(ny):
            point = h*np.array([idx1,idx2])
            min_dist = np.min((point[0],point[1], width-point[0], height-point[1]))
            value = hl*rl*(1/la.norm(point-posl)) + hr*rr*(1/la.norm(point-posr))
            #value = 2
            if la.norm(point-posl)>=rl and la.norm(point-posr)>=rr:
                X[ny*idx1+idx2,:] = np.array([point[0],point[1], min_dist*value])

    return X[X[:,0]<np.inf]

#def make_fo(width=10, height=5,rl=1,rr=1, hl=1,hr=-1, posl=[2.5,2.5], posr=[7.5,2.5], h=0.2):
#    def get_h(point):
#        point=np.array(point)
#        min_dist = np.min((point[0],point[1], width-point[0], height-point[1]))
#        value = hl*rl*(1/la.norm(point-posl)) + hr*rr*(1/la.norm(point-posr))
#        return min_dist*value
#    n=10
#    max = np.max([width,height])
#    Rl = np.square(np.linspace(rl,max,n))
#    Rr = np.square(np.linspace(rr,max,n))
#    X1 = np.linspace(0, 2*pi,n)
#    circles1 = np.array([[(x2*cos(x1)+posl[0], x2*sin(x1)+posl[1]) for x1 in X1]for x2 in Rl])
#    circles2 = np.array([[(x2*cos(x1)+posr[0], x2*sin(x1)+posr[1]) for x1 in X1]for x2 in Rr])
#    circle=np.vstack([np.vstack(circles1),np.vstack(circles2)])
#    X = []
#    for point in circle:
#        X.append([point[0],point[1], get_h(point)])
#    return np.array(X)
#





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

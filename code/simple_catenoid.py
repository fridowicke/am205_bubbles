#In this program we calculate the minimal surface Area of a Revolution
#from a point (x_1,y_1) to a point  (x_2,y_2) revolved around the x-axis
#The code is based on the Mathworld post https://mathworld.wolfram.com/MinimalSurfaceofRevolution.html
#Author: Friedrich Wicke
from math import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def get_curve(X1=[0,1],X2=[1,1],a_init=0.5, b_init=0.5, X=np.linspace(0,1,1000), plot=False):
    #Initialize the points
    x1, y1 = X1
    x2, y2 = X2

    #function for the surface
    def f(x):
        return a*np.cosh((x-b)/a)

    #Error function
    def error(y, *args):
        print(args)
        a,b=y
        x1,y1,x2,y2=args
        return abs(y1-a*cosh((x1-b)/a))+abs(y2-a*cosh((x2-b)/a))

    #Find the values a and b
    a,b=minimize(error,[a_init,b_init],(x1,y1,x2,y2)).x
    #print(error([a,b],*(x1,y1,x2,y2)))

    #Plotting the resulting Curve
    Y=f(X)

    if plot:
        plt.plot(X,Y)
        plt.show()

    return Y

def get_min_surface(h=1,r=1, n_samples=100000000):
    X1=[0,r]
    X2=[h,r]
    X = np.linspace(0,h,n_samples)
    vals = get_curve(X1=X1,X2=X2,X=X)
    sum = np.sum(vals)
    surface = 2*pi*sum*h/n_samples
    return surface

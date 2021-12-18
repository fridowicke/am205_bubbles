import make_shapes
import numpy as np
from math import *
from scipy.special import cotdg as cot
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.optimize import minimize
import plot_povray
from simple_catenoid import get_curve

#Function to compute the angle between three points
#In : Three points X=[x,y,z]
#Out: Angle
def angle(X):
    a,b,c = X
    b1 = a - b
    b2 = c - b
    b1 = b1 / la.norm(b1)
    b2 = b2 / la.norm(b2)
    dot_prod = np.dot(b1, b2)
    return np.arccos(dot_prod)


#Compute the approximation
def compute_grad(X):
    #inner function to order the points
    def order_clockwise(points, point_idx, idcs):
        point=points[point_idx]
        points_circle = np.hstack(np.array([]))
        pairs =

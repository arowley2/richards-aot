'''
Author: Amanda Rowley
Last Updated: 08/13/2024

Functions defining Van Genuchten soil hydraulic functions - each takes in a vector (either a Fenics vector type or numpy array).
'''

from fenics import *
from dolfin import *
import numpy as np
import math

from soil_params import get_soil_params


params = get_soil_params()

theta_r = params[0]
theta_s = params[1]
n = params[2]
alpha = params[3]
K_s = params[4]
m = params[5]
tau = params[6]

# Definition of Van Genuchten Soil Hydraulic Functions =============================================================================
def theta(h, V, nx):

    try:
        h_vec = h.vector().get_local()
    except:
        h_vec = h

    theta = np.zeros(nx+1)
    for i in range(nx+1):
        if h_vec[i] >= 0:
            theta[i] = theta_s
        else:
            theta[i] = theta_r + (theta_s - theta_r)/((1+np.abs(alpha*h_vec[i])**n)**m)

    return theta

def C_funct(h, V, nx):

    h_vec = h.vector().get_local()
    C = np.zeros(nx+1)
    for i in range(nx+1):
        if h_vec[i] >= 0:
            C[i] = 0
        else:
            C[i] = m*n*alpha*(theta_s - theta_r)*(np.abs(alpha*h_vec[i])**(n-1))/((1+np.abs(alpha*h_vec[i])**n)**(m+1))

    return C

def K_funct(h, V, nx):

    theta_vec = theta(h, V, nx)
    K = np.zeros(nx+1)
    for i in range(nx+1):
        Se = (theta_vec[i] - theta_r)/(theta_s - theta_r)
        K[i] = K_s*(Se**tau)*(1-(1-Se**(1/m))**m)**2
    
    return K


def h_funct(theta, V, nx):

    try:
        theta_vec = theta.vector().get_local()
    except:
        theta_vec = theta

    h = np.zeros(nx+1)
    for i in range(nx+1):
        if theta_vec[i] >= theta_s:
            h[i] = 0
        else:
            h[i] = -((((theta_s-theta_r)/(theta_vec[i] - theta_r))**(1/m) - 1)**(1/n))/(alpha)

    return h
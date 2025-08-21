'''
Author: Amanda Rowley
Last Updated: 03/07/2024

Functions for defining different soil types.
'''

from fenics import *
from dolfin import *

# Function called to get soil parameters - Default: sandy_loam_VG(), may change to any other pre-programmed soil type, or custom_VG()
def get_soil_params():
    params = sandy_loam_VG()
    return params

# Function for definining custom Van Genuchten soil params. Note: units are cm and minutes.
def custom_VG():

    theta_r = 0.065
    theta_s = 0.41
    n = 1.89
    alpha = 0.075
    K_s = 0.00322
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

# Pre-programmed soil parameters for Van Genuchten functions =========================================================================
def sand_VG():

    theta_r = 0.045
    theta_s = 0.43
    n = 2.68
    alpha = 0.145
    K_s = 0.495
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

def sandy_loam_VG():

    theta_r = 0.065
    theta_s = 0.41
    n = 1.89
    alpha = 0.075
    K_s = 0.0736806
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

def loamy_sand_VG():

    theta_r = 0.057
    theta_s = 0.41
    n = 2.28
    alpha = 0.124
    K_s = 0.243194
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

def loam_VG():
    
    theta_r = 0.078
    theta_s = 0.43
    n = 1.56
    alpha = 0.036
    K_s = 0.0173333
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

def silty_clay_VG():
    
    theta_r = 0.07
    theta_s = 0.36
    n = 1.09
    alpha = 0.005
    K_s = 0.000333333
    m = 1-(1/n)
    tau = 0.5
    
    return [theta_r, theta_s, n, alpha, K_s, m, tau]

'''
Author: Amanda Rowley
Last Updated: 08/13/2025

Functions for supporting data assimilation schemes - includes:
-- get_aot_params(), which can be adjusted for different numbers and frequencies of node-type observations, as well as interpolation constant mu.
-- feedback(), which can be adjusted to use different types of interpolation
'''

from fenics import *
from dolfin import *
import numpy as np
import math
import time
from scipy.interpolate import interp1d

from soil_hydraulics_VG import theta
from solve_support import Simpsons


## DA Support functions ==========================================================================================================================================

def get_actual(t, model_time, h_arr):

    model_time = np.asarray(model_time)
    h_arr = np.asarray(h_arr)
    model_time[model_time > t] = -1
    idx = (np.abs(model_time - t/1440)).argmin()

    return h_arr[idx]


def get_aot_params(dt, i=-1, j=-1, norm_diff = 1):

    collect_time = 60                                                # How often (in min) data is observed (can not be smaller than iterating timestep)
    times = [60, 30, 5, 1]
    if (i >= 0):
        collect_time = times[i]
    
    num_nodes = 7                                                   # How many nodes are observed observed (can not be more than nx+1, must be positive integer)
    nodes = [5, 7, 10, 12]
    if (j >= 0):
        num_nodes = nodes[j]

    # crns_collect_time = 60
    # remote_collect_time = 1440

    mu = (1.8/dt)/20

    return mu, collect_time, num_nodes


# Determine whether to observe data for interpolation.
def get_obs(t, prev_obs, collect_time):
    if (prev_obs + collect_time <= t):
        return True, t
    else:
        return False, prev_obs


def get_integrated_top(vec, top_cm, nx, min_x, max_x):
    top_layer = np.array(vec)
    delta_x = (max_x - min_x)/nx
    top_layer = top_layer[int((-min_x-top_cm)*delta_x):]
    integrated_val = Simpsons(int(top_cm/delta_x), min_x, min_x+top_cm, top_layer)
    return integrated_val


def Ih_integrated(theta, V, top_cm, nx, min_x, max_x):

    integrated_val = get_integrated_top(theta, top_cm, nx, min_x, max_x)
    integral_avg = integrated_val/(top_cm)

    top_layer = np.zeros(nx+1)

    delta_x = (max_x - min_x)/nx
    # # Option 1: Set midpoint of observed layer to average integrated value.
    # top_layer[int(top_cm/2)] = integral_avg

    # Option 2: Set all nodes in observed layer to average integrated value.
    top_layer[int((-min_x-top_cm)*delta_x):] = integral_avg

    return top_layer

## Interpolation Functions =============================================================================================================

# Remote Sensing (set collect_time = 1440, 2880, or 4320 min):
# top_cm = 5
def Ih_Remote(theta, V, nx, min_x, max_x):
    theta_interp = Ih_integrated(theta, V, 5, nx, min_x, max_x)
    return theta_interp

# Cosmic-Ray Neutron Sensor (set collect_time = 30 or 60 min):
# top_cm = 30
def Ih_CRNS(theta, V, nx, min_x, max_x):
    theta_interp = Ih_integrated(theta, V, 30, nx, min_x, max_x)
    return theta_interp

# Models point sensors at specifically placed nodes in conjunction with CRNS and/or Remote sensing data, depending on crns_bool and remote_bool
def Ih_combined(theta, V, num_nodes, crns_bool, remote_bool, nx, min_x, max_x):
    
    pointwise_interp, obs_points = Ih_piecewise_spec(theta, V, num_nodes, min_x, nx)

    pointwise_top_30 = get_integrated_top(pointwise_interp, 30, nx, min_x, max_x)
    pointwise_top_5 = get_integrated_top(pointwise_interp, 5, nx, min_x, max_x)

    theta_interp = pointwise_interp.copy()
    delta_x = (max_x - min_x)/nx

    if remote_bool:
        remote_top_5 = get_integrated_top(theta, 5, nx, min_x, max_x)
        diff_in_5 = remote_top_5 - pointwise_top_5

        remote_count = 0
        for i in range(int((-min_x-5)*delta_x), -min_x+1, 1):
            if not (i in (-obs_points)):
                remote_count = remote_count + 1

        if remote_count > 0:
            remote_diff_avg = diff_in_5/5
            for i in range(int((-min_x-5)*delta_x), -min_x+1, 1):
                if not (i in (-obs_points)):
                    theta_interp[i] = theta_interp[i] + remote_diff_avg

        if crns_bool:
            crns_top_30 = get_integrated_top(theta, 30, nx, min_x, max_x)
            crns_remain = crns_top_30 - remote_top_5
            diff_in_30 = crns_remain - (pointwise_top_30 - pointwise_top_5)

            crns_count = 0
            for i in range(int((-min_x-25)*delta_x), -min_x+1, 1):
                adj_i = i - 6
                if not ( adj_i in (-obs_points)):
                    crns_count = crns_count + 1

            if crns_count > 0:
                crns_diff_avg = diff_in_30/25
                for i in range(int((-min_x-25)*delta_x), -min_x+1, 1):
                    adj_i = i - 6
                    if not (adj_i in (-obs_points)):
                        theta_interp[adj_i] = theta_interp[adj_i] + crns_diff_avg

    elif crns_bool:
        crns_top_30 = get_integrated_top(theta, 30, nx, min_x, max_x)
        diff_in_30 = crns_top_30 - pointwise_top_30
    
        crns_count = 0
        for i in range(int((-min_x-30)*delta_x), -min_x+1, 1):
            if not ( i in (-obs_points)):
                crns_count = crns_count + 1

        if crns_count > 0:
            crns_diff_avg = diff_in_30/30
            for i in range(int((-min_x-30)*delta_x), -min_x+1, 1):
                if not (i in (-obs_points)):
                    theta_interp[i] = theta_interp[i] + crns_diff_avg

    theta_interp_funct = Function(V)
    theta_interp_funct.vector().set_local(theta_interp)
    return theta_interp

# Models point sensors with information at specific number of uniformly spaced nodes
def Ih_piecewise_uni(theta, V, num_nodes, min_x, nx):

    low = 0
    high = min_x
    
    x_interp = -np.linspace(-low, -high, num=num_nodes, dtype=float)
    x_indices = np.floor((-nx/min_x)*(-x_interp)).astype(int)
    y_interp = theta[x_indices]

    x_vals = Expression('x[0]', degree=1)
    x_vals = project(x_vals, V)
       
    f_interp = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')     
    y_vals = (f_interp(x_vals.vector().get_local()))

    return y_vals

# Models point sensors with information at specific number of uniformly spaced nodes throughout top half of column
def Ih_piecewise_uni_top(theta, V, num_nodes, min_x, nx):

    low = 0
    high = min_x

    x_interp = -np.linspace(-int(high/2), -high, num=num_nodes, dtype=float)
    x_interp = np.insert(x_interp, 0, 0)
    x_indices = np.floor((-nx/min_x)*(-x_interp)).astype(int)

    y_interp = theta[x_indices]

    x_vals = Expression('x[0]', degree=1)
    x_vals = project(x_vals, V)
       
    f_interp = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')     
    y_vals = (f_interp(x_vals.vector().get_local()))

    return y_vals
    

# Models point sensors with information at specific number of nodes at specified depths
def Ih_piecewise_spec(theta, V, num_nodes, min_x, nx):

    # This allows for selection of specific, non-uniformly spaced depths: (Default: Depths 5, 10, 25, 50, 100 cm)
    x_interp = np.array([-100, -150, -175, -190, -195])
    x_indices = np.floor((-nx/min_x)*(-x_interp)).astype(int)
    
    y_interp = theta[x_indices]

    x_vals = Expression('x[0]', degree=1)
    x_vals = project(x_vals, V)
       
    f_interp = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')     
    y_vals = (f_interp(x_vals.vector().get_local()))

    return y_vals, x_indices

# Models point sensors with information at specific number of uniformly spaced nodes and zeros everywhere else
def Ih_zeros(theta, V, num_nodes, min_x, nx):

    low = 0
    high = min_x
    
    x_interp = -np.linspace(-low, -high, num=num_nodes, dtype=float)
    x_indices = np.floor((-nx/min_x)*(-x_interp)).astype(int)
    
    theta_interp = np.zeros((len(theta)))
    for x in x_indices:
        theta_interp[x] = theta[x]
    
    return theta_interp

# Function for returning interpolated vectors - May use to alter type of interpolation ================================================================
def feedback(h_m, h_m_dA, params, V, nx, min_x, max_x, num_nodes, remote_bool=False, crns_bool=False):

    # Ih_theta_dA = Ih_piecewise_uni(theta(h_m_dA, V, nx), V, num_nodes, min_x, nx)                                   # Observation type: nodes at uniform depths, linear interpolation between nodes
    # Ih_theta = Ih_piecewise_uni(theta(h_m, V, nx), V, num_nodes, min_x, nx)

    # Ih_theta_dA, _ = Ih_piecewise_spec(theta(h_m_dA, V, nx), V, num_nodes, min_x, nx)                             # Observation type: nodes at specified depths, linear interpolation between nodes
    # Ih_theta, _ = Ih_piecewise_spec(theta(h_m, V, nx), V, num_nodes, min_x, nx)

    # Ih_theta_dA = Ih_zeros(theta(h_m_dA, V, nx), V, num_nodes, min_x, nx)                                         # Observation type: nodes at uniform depths, zeros filled in between nodes
    # Ih_theta = Ih_zeros(theta(h_m, V, nx), V, num_nodes, min_x, nx)

    Ih_theta_dA = Ih_combined(theta(h_m_dA, V, nx), V, num_nodes, crns_bool, remote_bool, nx, min_x, max_x)       # Observation type: combination of nodes and CRNS and/or satellite (determined by values passed by main.py)
    Ih_theta = Ih_combined(theta(h_m, V, nx), V, num_nodes, crns_bool, remote_bool, nx, min_x, max_x)

    # Ih_theta_dA = Ih_CRNS(theta(h_m_dA, V, nx), V, nx, min_x, max_x)                                              # Observation type: CRNS (integrated over top 30 cm) data, zeros elsewhere
    # Ih_theta = Ih_CRNS(theta(h_m, V, nx), V, nx, min_x, max_x)
    
    # Ih_theta_dA = Ih_Remote(theta(h_m_dA, V, nx), V, nx, min_x, max_x)                                            # Observation type: Remote (integrated over top 5 cm) data, zeros elsewhere
    # Ih_theta = Ih_Remote(theta(h_m, V, nx), V, nx, min_x, max_x)


    return Ih_theta, Ih_theta_dA

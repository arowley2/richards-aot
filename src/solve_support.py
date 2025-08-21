'''
Author: Amanda Rowley
Last Updated: 08/19/2024

Functions for solving Richards using finite difference methods, as well as for plotting and saving results.
Also contains miscellaneous functions for approximating mass balance.
'''

from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import math
from scipy.io import savemat
import os
import time
import shutil

from boundaries import get_type_bounds
from soil_hydraulics_VG import K_funct, C_funct, theta
from soil_params import get_soil_params

top_bound, bottom_bound = get_type_bounds()
settime = time.time()
curtime = settime

## Functions to support iterated solver ===================================================================================================================

# Function to solve Richards Equation using Finite Difference Methods as described in 1D Hydrus Manual

def solver(h_n, h_m, theta_n, theta_m, C_m, K_m, V, nx, delta_x, dt, g_top, gt_top, g_bottom, interpt=False, mu=0, Ih_theta=[], Ih_theta_dA=[]):    

    theta_n.vector().set_local(theta(h_n, V, nx))
    theta_m.vector().set_local(theta(h_m, V, nx))
    C_m.vector().set_local(C_funct(h_m, V, nx))
    K_m.vector().set_local(K_funct(h_m, V, nx))

    P = np.zeros((nx+1,nx+1))
    F = np.zeros(nx+1)

    for i in range(nx+1):

        if i==0: # Bottom node

            if bottom_bound == 0: # Dirichlet
                d = 1
                e = -(K_m.vector().get_local()[i+1] + K_m.vector().get_local()[i])/(2*delta_x)
                f = h_m.vector().get_local()[0]
            else:
                d = (K_m.vector().get_local()[0] + K_m.vector().get_local()[1])/(2*delta_x)
                e = -(K_m.vector().get_local()[i+1] + K_m.vector().get_local()[i])/(2*delta_x)
                f = (K_m.vector().get_local()[i+1] + K_m.vector().get_local()[i])/(2) + float(g_bottom)

            P[i,i] = d
            P[i,i+1] = e
            P[i+1,i] = e
            F[i] = f

        elif i==nx: # Top node

            if top_bound == 0: # Dirichlet
                d = 1
                f = h_m.vector().get_local()[nx]
            else:
                d = (delta_x/(2*dt))*(C_m.vector().get_local()[i]) + (K_m.vector().get_local()[nx] + K_m.vector().get_local()[nx-1])/(2*delta_x)
                f = (delta_x/(2*dt))*(C_m.vector().get_local()[i])*(h_m.vector().get_local()[i]) - (delta_x/(2*dt))*((theta_m.vector().get_local()[i])-(theta_n.vector().get_local()[i])) - (K_m.vector().get_local()[nx] + K_m.vector().get_local()[nx-1])/(2) - float(g_top)
            
                if interpt:
                    f = f + mu*(Ih_theta-Ih_theta_dA)[i]*delta_x/2

            P[i,i] = d
            F[i] = f

        else:
            d = (delta_x/dt)*(C_m.vector().get_local()[i]) + (K_m.vector().get_local()[i+1] + K_m.vector().get_local()[i])/(2*delta_x) + (K_m.vector().get_local()[i-1] + K_m.vector().get_local()[i])/(2*delta_x)
            e = -(K_m.vector().get_local()[i+1] + K_m.vector().get_local()[i])/(2*delta_x)
            f = (delta_x/dt)*(C_m.vector().get_local()[i])*(h_m.vector().get_local()[i]) - (delta_x/dt)*((theta_m.vector().get_local()[i])-(theta_n.vector().get_local()[i])) + (K_m.vector().get_local()[i+1] - K_m.vector().get_local()[i-1])/(2)
            
            if (nx-i)*delta_x <= 50:
                f = f - float(gt_top)*delta_x
            if interpt:
                f = f + mu*(Ih_theta-Ih_theta_dA)[i]*delta_x

            P[i,i] = d
            P[i,i+1] = e
            P[i+1, i] = e
            F[i] = f

        if top_bound == 0:
            P[nx, nx-1] = 0
        if bottom_bound == 0:
            P[0,1] = 0

    h = solve(P,F) 

    theta_before = theta(h, V, nx)
    h[h < -(10**5)] = -(10**5)
    h[h > -(0.1)] = -(0.1)

    theta_after = theta(h, V, nx)

    return h

## Functions for plotting ========================================================================================================================

# Function to initialize arrays for plotting and data storage
def plot_initial(h_m, V, nx):
    model_time = []
    cpu_time = []

    cpu_time.append(0)
    model_time.append(0)

    mass_balance_arr = []
    mass_balance_arr.append(1)
    mass_loss_arr = []
    mass_loss_arr.append(0)

    theta_arr = []
    h_arr = []

    theta_arr.append((theta(h_m, V, nx)))
    h_arr.append((h_m.vector().get_local()))

    return model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr

# Function to initialize arrays for data assimilation plotting and data storage
def plot_dA_initial(h_m_dA, h_m, V, nx, Ih_theta, Ih_theta_dA):
    diff_norm = []
    diff_norm.append(np.linalg.norm(theta(h_m, V, nx) - theta(h_m_dA, V, nx)))

    diff_arr = []
    diff_arr.append(theta(h_m,V,nx)-theta(h_m_dA, V, nx))

    Ih_act = []
    Ih_dA = []
    Ih_act.append(Ih_theta)
    Ih_dA.append(Ih_theta_dA)
    
    return diff_norm, diff_arr, Ih_act, Ih_dA


# Function to update arrays for plotting and data storage
def plot_update(model_time, cpu_time, theta_arr, h_arr, start_time, t, h_m, V, nx, mass_balance_arr = [], mass_loss_arr = [], mass_loss = 0, mass_balance = 0 ):

    current_time = time.process_time()
    cpu_time.append((current_time - start_time)/60)
    model_time.append(t/1440)
    
    theta_arr.append((theta(h_m, V, nx)))
    h_arr.append((h_m.vector().get_local()))

    mass_balance_arr.append(mass_balance)
    mass_loss_arr.append(mass_loss)
    

    return model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr

# Function to update arrays for data assimilation plotting and data storage
def plot_dA_update(diff_norm, diff_arr, Ih_act, Ih_dA, h_m_dA, h_m, V, nx, Ih_theta, Ih_theta_dA):
    diff_norm.append(np.linalg.norm(theta(h_m, V, nx) - theta(h_m_dA, V, nx)))
    diff_arr.append(theta(h_m,V,nx)-theta(h_m_dA, V, nx))

    Ih_act.append(Ih_theta)
    Ih_dA.append(Ih_theta_dA)
    
    return diff_norm, diff_arr, Ih_act, Ih_dA


def plot_save(model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr, min_x, max_x, nx):
    
    filepath = os.path.dirname(__file__)
    res = filepath.rsplit('/',1)
    workdir = res[0]
    
    workdir = workdir + '/results'
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    namestring = '_actual'

    theta_dict = {"model_time" : model_time, "cpu_time" : cpu_time, "saturation" : theta_arr, "pressure" : h_arr, "mass_balance" : mass_balance_arr, "mass_loss" : mass_loss_arr}
    savemat(workdir + '/theta_dict' + namestring + '.mat', theta_dict)

    # Plotting theta vs depth over time
    x = np.linspace(min_x, max_x, nx+1)
    for theta_in_arr in theta_arr:
        plt.plot((theta_in_arr),x)
    plt.ylabel('Actual Depth (cm)')
    plt.xlabel('Water Content')
    plt.grid()
    plt.savefig(workdir + '/theta_vs_depth' + namestring + '.png')

    plt.close()

    # Plotting mass balance over time
    plt.plot(np.array(model_time), np.array(mass_balance_arr))
    plt.ylabel('Mass Balance')
    plt.xlabel('Time')
    plt.savefig(workdir + '/mass_balance' + namestring + '.png')

    plt.close()

    # Plotting percent mass loss over time
    plt.plot(np.array(model_time), np.array(mass_loss_arr))
    plt.ylabel('Percent Mass Loss')
    plt.xlabel('Time')
    plt.savefig(workdir + '/mass_loss' + namestring + '.png')

    plt.close()


def plot_dA_save(model_time, cpu_time, theta_arr, h_arr, diff_norm_at_time, collect_time, num_nodes, min_x, max_x, nx, diff_arr, Ih_act, Ih_dA, type=0):


    filepath = os.path.dirname(__file__)
    res = filepath.rsplit('/',1)
    workdir = res[0]
    
    workdir = workdir + '/results'
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
        

    if collect_time < 1:
        snew = str(collect_time).replace('.', 'p')
        namestring = '_dt' + snew + '_nodes' + str(num_nodes)
    else:
        namestring = '_dt' + str(collect_time) + '_nodes' + str(num_nodes)
    
    if type==1:
        namestring = namestring + '_Kalman'
    elif type==2:
        namestring = namestring + '_Nudging'
    else:
        namestring = namestring + '_AOT'


    filename = '/theta_dA_dict' + namestring + '.mat'

    theta_dict = {"model_time" : model_time, "cpu_time" : cpu_time, "approx_saturation" : theta_arr, "approx_pressure" : h_arr, "norm_diff" : diff_norm_at_time, "diff_arr" : diff_arr, "Ih_act" : Ih_act, "Ih_dA" : Ih_dA}
    savemat(workdir + filename, theta_dict)

    # Plotting theta vs depth over time
    x = np.linspace(min_x, max_x, nx+1)
    for theta_in_arr in theta_arr:
        plt.plot((theta_in_arr),x)
    plt.ylabel('Approx Depth (cm)')
    plt.xlabel('Water Content')
    plt.grid()
    plt.savefig(workdir + '/theta_dA_vs_depth' + namestring + '.png')

    plt.close()

    # Plotting RMSE of water content
    plt.semilogy(model_time, diff_norm_at_time)
    plt.xlabel('Time (in days)')
    plt.ylabel('Norm of theta - theta_dA')
    plt.savefig(workdir + '/norm_diff' + namestring + '.png', bbox_inches='tight')

    plt.close()


def savefiles():
    
    # Save src code ============================================================

    filepath = os.path.dirname(__file__)
    res = filepath.rsplit('/',1)
    workdir = res[0]
    
    srcdir = workdir + '/src'
    
    workdir = workdir + '/results'
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
        
    workdir = workdir + '/' + str(int(curtime))
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
        
    newsrcdir = workdir + '/src'
    shutil.copytree(srcdir, newsrcdir)
    

# # Functions for calculating mass ==========================================================================

# Function to calculate mass balance and percent mass error

def mass_calc(h_m, V, nx, max_x, min_x, delta_x, dt, total_flux, theta_0):

    # Calculate mass balance ==============================================

    dh_N = (h_m.vector().get_local())[nx]-(h_m.vector().get_local())[nx-1]
    dh_1 = (h_m.vector().get_local())[1]-(h_m.vector().get_local())[0]
    
    K_theta_N_min_1 = (K_funct(h_m, V, nx))[nx-1]
    K_theta_N = (K_funct(h_m, V, nx))[nx]
    K_theta_0 = (K_funct(h_m, V, nx))[0]
    K_theta_1 = (K_funct(h_m, V, nx))[1]
    
    flux_N = ((K_theta_N_min_1+K_theta_N)/2)*(dh_N/delta_x + 1)*dt
    flux_1 = ((K_theta_0+K_theta_1)/2)*(dh_1/delta_x + 1)*dt

    total_flux = total_flux + (flux_N - flux_1)

    theta_at_time = theta(h_m, V, nx)
    sliced_theta_0 = theta_0[1:-1]
    sliced_theta_n = theta_at_time[1:-1]
    added_mass = delta_x*np.sum(sliced_theta_n - sliced_theta_0)
    mass_balance = added_mass/total_flux

    # Calculate mass loss =================================================
    mass_change = Simpsons(nx, min_x, max_x, sliced_theta_n, 2) - Simpsons(nx, min_x, max_x, sliced_theta_0,2 )
    expected_mass = Simpsons(nx, min_x, max_x, sliced_theta_0, 2) + total_flux
    actual_mass = Simpsons(nx, min_x, max_x, sliced_theta_n, 2)

    mass_percent_err = abs((expected_mass - actual_mass)/actual_mass)*100
    mass_ratio = abs(actual_mass/expected_mass)

    return total_flux, mass_balance, mass_percent_err



# Simpsons function approximates an integral given a finite number of points (used in mass calculations).

def Simpsons(nx, min_x, max_x, theta=[], adj=0):

    i = 1
    upper = ((nx-adj)/2)
    h = (max_x - min_x - adj)/(nx-adj)
    summation = 0

    while (i <= upper):
        summation = summation + theta[(2*i-2)] + 4*theta[(2*i-1)] + theta[2*i]
        i = i+1

    return (h/3)*summation


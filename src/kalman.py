'''
Author: Amanda Rowley
Last Updated: 08/19/2024

Function for approximating solution with EnKF, following the algorithm given by Asch, Bocquet, and Nodet in
Data Assimilation: Methods, Algorithms, and Applications. May want to change parameters gamma, sigma, delta,
which relate to the standard deviation of the gaussian noise in the model, observations, and inflation.
'''

from fenics import *
from dolfin import *
import numpy as np
from scipy.linalg import solve
import math
from tqdm import tqdm
import time

from boundaries import get_bounds
from soil_hydraulics_VG import theta, h_funct
from soil_params import get_soil_params
from solve_support import *
from da_support import *


# Function to approximate Richards Equation solution using Ensemble Kalman Filter
def kalman_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0, tol, max_iter, actual_model_time, h_actual_arr, theta_arr, collect_time, num_nodes, num_ensembles):

    # Initialization =======================================================================

    def_dt = dt
    current_day = 1

    theta_r = params[0]
    theta_s = params[1]
    n = params[2]
    alpha = params[3]
    K_s = params[4]
    m = params[5]
    tau = params[6]

    K_m = Function(V)
    C_m = Function(V)
    theta_m = Function(V)
    theta_n = Function(V)
    h_m = Function(V)
    h_n = Function(V)

    theta_temp = Function(V)
    h_temp = Function(V)

    g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)

    h_act = get_actual(0, actual_model_time, h_actual_arr)

    # Plotting ==================================================================

    Ih_theta, Ih_theta_dA = feedback(h_act, h_m, params, V, nx, min_x, max_x, num_nodes)
    model_time, cpu_time, _, _, theta_arr, h_arr = plot_initial(h_m, V, nx)
    diff_norm, diff_arr, Ih_act, Ih_dA = plot_dA_initial(h_m, h_act, V, nx, Ih_theta, Ih_theta_dA)

    start_time = time.process_time()

    # Ensemble Parameters =======================================================

    gamma = 1e-12                                   # Ensemble generated noise standard deviation
    sigma = 1e-12                                   # Observation noise standard deviation
    delta = 1e-12                                   # Inflation noise standard deviation

    K = num_ensembles                               # Number of ensembles

    # Initialization
    va = np.zeros((1, nx+1, K))
    vf = np.zeros((1, nx+1, K))
    vf_temp = np.zeros((1, nx+1, K))
    u_err = np.zeros((nx+1, K))
    mf = np.zeros((1, nx+1))
    mu_err = np.zeros((1, nx+1))
    mv = np.zeros((1, nx+1))
    y = np.zeros((1, nx+1, K))

    h_act = get_actual(0, actual_model_time, h_actual_arr)  

    h_n.assign(project(h_0, V))
    h_m.assign(project(h_0, V))

    # Initialize the ensemble
    for k in range(K):
        u_err[:,k] = gamma*np.random.randn(nx+1)
        vf[0,:,k] = np.array(h_m.vector().get_local()) + 0.01*np.random.randn(1)
        vf[0,:,k] = np.clip(vf[0,:,k], -1e5, -1e-1)
        va[0,:,k] = vf[0,:,k]

    avg_va = np.sum(va[-1, :, :], axis=1)/K

    # Plotting ==================================================================

    start_time = time.process_time()

    # Iterated Solver ============================================================================================================================

    t = 0
    prev_obs = -1000

    quit_early = False

    pbar = tqdm(total = T) 
    while t < T:
        avg_eps = 0 
        avg_iter_m = 0

        time_reduce = 0
        force_reduce = 0
        no_force = False

        g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)

        theta_act = theta(get_actual(t, actual_model_time, h_actual_arr), V, nx)
        h_act = get_actual(t, actual_model_time, h_actual_arr)

        obs = Ih_piecewise_uni(h_act, V, num_nodes, min_x, nx)

        for k in range(K):
            h_temp.vector().set_local(va[0,:,k])
            h_m.vector().set_local(h_temp.vector().get_local())
            h_n.vector().set_local(h_temp.vector().get_local())

            eps = 100000 
            iter_m = 0

            while eps > tol and iter_m < max_iter:

                # Update of the ensemble forecast
                h = solver(h_n, h_m, theta_n, theta_m, C_m, K_m, V, nx, delta_x, dt, g_top, gt_top, g_bottom)
                vf_temp[0,:,k] = h
               
                # If no overflow, continue with Picard iteration scheme
                if all(i >= 0 for i in (h*(h_m.vector().get_local()))):
                    diff = h_m.vector().get_local() - h
                    h_m.vector().set_local(h)

                    eps = np.linalg.norm(diff, ord=np.Inf)
                    iter_m = iter_m + 1
                else:
                    print('Something went wrong.')
                    quit()

            if eps > tol:
                print('Picard error exceeds tolerance: ', eps)

            avg_iter_m = iter_m + avg_iter_m
            avg_eps = eps + avg_eps
        
        avg_iter_m = avg_iter_m/K
        avg_eps = avg_eps/K
        if avg_eps < tol:

            t = t + dt

            current_day = math.ceil(t/1440)
            pbar.update(dt)

        else:
            print('No converge, time: ', dt)
            break

        theta_act = theta(get_actual(t, actual_model_time, h_actual_arr), V, nx)
        h_act = get_actual(t, actual_model_time, h_actual_arr)


        interpt, prev_obs = get_obs(t-dt, prev_obs, collect_time)

        if interpt:
        
            obs = Ih_piecewise_uni(h_act, V, num_nodes, min_x, nx)
    
            # Inflate the ensemble forecast:
            for k in range(K):
                # Inflate the ensemble forecast:
                vf[0,:,k] = vf_temp[0,:,k] + delta*np.random.randn(nx+1)
                vf[0,:,k] = np.clip(vf[0,:,k], -1e5, -1e-2)
    
                # Generate ensemble observations
                u_err[:, k] = gamma*np.random.randn(nx+1)
                y[0,:,k] = obs + sigma*(np.array(np.random.randn(1))) + u_err[:,k]
                y[0,:,k] = np.clip(y[0,:,k], -1e5, -1e-2)
            
            # Compute ensemble means
            mf[0,:] = np.sum(vf[0, :, :], axis=1)/K
            mu_err[0,:] = np.sum(u_err, axis=1)/K
            mv[0,:] = Ih_piecewise_uni(mf[0,:], V, num_nodes, min_x, nx)
    
            # Compute normalized anomalies
            W = np.zeros((nx+1,K))
            U = np.zeros((nx+1,K))
            for k in range(K):
                W[:,k] = (vf[0,:,k] - mf[0,:])/sqrt(K-1)
                U[:,k] = (Ih_piecewise_uni(vf[0,:,k], V, num_nodes, min_x, nx) - u_err[:,k] - mv[0,:] + mu_err[0,:])/sqrt(K-1)
            
            # Compute the gain
            A = W@np.transpose(U)
            B = U@np.transpose(U)
            matK = np.transpose(np.linalg.solve(np.transpose(B), np.transpose(A)))
    
            # Update of the ensemble
            for k in range(K):
                va[0,:,k] = vf[0,:,k] + (matK@(y[0,:,k] - Ih_piecewise_uni(vf[0,:,k], V, num_nodes, min_x, nx)))
                va[0,:,k] = np.clip(va[0,:,k], -1e5, -1e-2)
            
        
        else:
            for k in range(K):
                va[0,:,k] = vf_temp[0,:,k]
                va[0,:,k] = np.clip(va[0,:,k], -1e5, -1e-2)


        avg_va = np.sum(va[-1, :, :], axis=1)/K

        h_act = get_actual(t, actual_model_time, h_actual_arr)

        model_time, cpu_time, _, _, theta_dA_arr, h_dA_arr = plot_update(model_time, cpu_time, theta_arr, h_arr, start_time, t, h_m, V, nx)
        diff_norm, diff_arr, Ih_act, Ih_dA = plot_dA_update(diff_norm, diff_arr, Ih_act, Ih_dA, avg_va, h_act, V, nx, Ih_theta, Ih_theta_dA)

    pbar.close()

    end_time = time.process_time()

    plot_dA_save(model_time, cpu_time, theta_arr, h_arr, diff_norm, collect_time, num_nodes, min_x, max_x, nx, diff_arr, Ih_act, Ih_dA, 1)

    print('Total cpu time (in minutes): ', (end_time - start_time)/60)


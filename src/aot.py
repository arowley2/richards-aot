'''
Author: Amanda Rowley
Last Updated: 08/19/2024

Function containing wrapper for AOT solver. Can be altered to incorporated adaptive timestepping.
'''

from fenics import *
from dolfin import *
import numpy as np
import math
import time
from tqdm import tqdm
from scipy.interpolate import interp1d

from boundaries import get_bounds
from soil_hydraulics_VG import K_funct, C_funct, theta
from soil_params import get_soil_params
from solve_support import *
from da_support import *

# Function to solve Richards Equation using Picard loop
def aot_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0, tol, max_iter, actual_model_time, h_actual_arr, theta_arr, collect_time, num_nodes, crns_collect_time=0, remote_collect_time=0):

    # Initialization =============================================================================================

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

    h_n.assign(project(h_0, V))
    h_m.assign(project(h_0, V))
    theta_0 = theta(h_m, V, nx)

    g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)

    h_act = get_actual(0, actual_model_time, h_actual_arr)

    # Plotting ===================================================================================================

    Ih_theta, Ih_theta_dA = feedback(h_act, h_m, params, V, nx, min_x, max_x, num_nodes)
    model_time, cpu_time, _, _, theta_arr, h_arr = plot_initial(h_m, V, nx)
    diff_norm, diff_arr, Ih_act, Ih_dA = plot_dA_initial(h_m, h_act, V, nx, Ih_theta, Ih_theta_dA)

    start_time = time.process_time()

    # Iterated Solver ================================================================================================

    t = 0

    prev_obs = -1000
    prev_remote = -1000
    prev_crns = -1000

    remote_bool = False
    crns_bool = False

    mu, _, _ = get_aot_params(dt)

    pbar = tqdm(total = T) 
    while t < T:
        eps = 10000000
        iter_m = 0

        g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)

        h_act = get_actual(t, actual_model_time, h_actual_arr)
        Ih_theta, Ih_theta_dA = feedback(h_act, h_m, params, V, nx, min_x, max_x, num_nodes, remote_bool=False, crns_bool=False)
        diff_for_mu = np.linalg.norm(Ih_theta - Ih_theta_dA)
        mu, _, _ = get_aot_params(dt, -1, -1, diff_for_mu)

        interpt, prev_obs = get_obs(t, prev_obs, collect_time)

        if crns_collect_time > 0:
            crns_bool, prev_crns = get_obs(t,prev_crns, crns_collect_time)
        if remote_collect_time > 0:
            remote_bool, prev_remote = get_obs(t, prev_remote, remote_collect_time)

        while eps > tol and iter_m < max_iter:

            if interpt:
                Ih_theta, Ih_theta_dA = feedback(h_act, h_m, params, V, nx, min_x, max_x, num_nodes, remote_bool, crns_bool)
            else:
                Ih_theta, Ih_theta_dA = None, None
            
            h = solver(h_n, h_m, theta_n, theta_m, C_m, K_m, V, nx, delta_x, dt, g_top, gt_top, g_bottom, interpt, mu, Ih_theta, Ih_theta_dA)

            diff = h_m.vector().get_local() - h
            h_m.vector().set_local(h)

            eps = np.linalg.norm(diff, ord=np.Inf)
            iter_m = iter_m + 1

        if eps < tol:
            h_m.vector().set_local(h)
            h_n.vector().set_local(h_m.vector().get_local())

            t = t + dt

            current_day = math.ceil(t/1440)
            pbar.update(dt)

            if iter_m <= 3:
                dt = def_dt
            
            # # Can be uncommented to increase efficiency, but worsens mass conservation
            # if iter_m >= 7:
            #     dt = dt*0.8

            h_act = get_actual(t, actual_model_time, h_actual_arr)
            Ih_theta, Ih_theta_dA = feedback(h_act, h_m, params, V, nx, min_x, max_x, num_nodes)

            model_time, cpu_time, _, _, theta_dA_arr, h_dA_arr = plot_update(model_time, cpu_time, theta_arr, h_arr, start_time, t, h_m, V, nx)
            diff_norm, diff_arr, Ih_act, Ih_dA = plot_dA_update(diff_norm, diff_arr, Ih_act, Ih_dA, h_m, h_act, V, nx, Ih_theta, Ih_theta_dA)
            
        else:
            h_n.vector().set_local(h_n.vector().get_local())
            h_m.vector().set_local(h_n.vector().get_local())

            print('No converge...')
            break

    pbar.close()
    end_time = time.process_time()
    print('Total cpu time (in minutes): ', (end_time - start_time)/60)

    plot_dA_save(model_time, cpu_time, theta_arr, h_arr, diff_norm, collect_time, num_nodes, min_x, max_x, nx, diff_arr, Ih_act, Ih_dA)

    return model_time, h_arr, theta_arr


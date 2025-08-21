'''
Author: Amanda Rowley
Last Updated: 08/19/2024

Function containing wrapper for solver. Can be altered to incorporated adaptive timestepping.
'''

from fenics import *
from dolfin import *
import numpy as np
import math
from tqdm import tqdm
import time

from boundaries import get_bounds
from soil_hydraulics_VG import theta
from soil_params import get_soil_params
from solve_support import *

# Function to solve Richards Equation using Picard loop
def iterated_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0, tol, max_iter):

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

    h_n.assign(project(h_0, V))
    h_m.assign(project(h_0, V))
    theta_0 = theta(h_m, V, nx)

    g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)

    # Plotting ==================================================================

    model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr = plot_initial(h_m, V, nx)
    start_time = time.process_time()

    total_flux = 0

    # Iterated Solver ============================================================================================================================

    t = 0

    quit_early = False

    pbar = tqdm(total = T) 
    while t < T:
        eps = 10000000
        iter_m = 0

        time_reduce = 0
        force_reduce = 0
        no_force = False

        g_top, gt_top, g_bottom = get_bounds(current_day, h_m, nx, max_x, min_x, params, V, dt)
        
        while eps > tol and iter_m < max_iter:

            h = solver(h_n, h_m, theta_n, theta_m, C_m, K_m, V, nx, delta_x, dt, g_top, gt_top, g_bottom)

            # If no overflow, continue with Picard iteration scheme
            if all(i >= 0 for i in ((h*(h_m.vector().get_local())))):
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

            total_flux, mass_balance, mass_loss = mass_calc(h_m, V, nx, max_x, min_x, delta_x, dt, total_flux, theta_0)
            model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr = plot_update(model_time, cpu_time, theta_arr, h_arr, start_time, t, h_m, V, nx, mass_balance_arr, mass_loss_arr, mass_loss, mass_balance)
            
        else:
            h_n.vector().set_local(h_n.vector().get_local())
            h_m.vector().set_local(h_n.vector().get_local())

            dt = dt/2

            print('No converge, reducing time: ', dt)

            if dt < dt_min:
                print('Did not converge.')
                break

    pbar.close()

    end_time = time.process_time()

    plot_save(model_time, cpu_time, mass_balance_arr, mass_loss_arr, theta_arr, h_arr, min_x, max_x, nx)

    print('Total cpu time (in minutes): ', (end_time - start_time)/60)


    return model_time, h_arr, theta_arr



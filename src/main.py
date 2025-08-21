'''
Author: Amanda Rowley
Last Updated: 08/19/2024

Main method for running AOT solver. Contains most parameters - others are in aot.py (specifically get_aot_params() and feedback()), global variables in boundaries.py, and soil_params.py (mainly get_soil_params()).
Contains calls to approximate solution with three types of data assimilation - AOT, Nudging, and Ensemble Kalman.
To run in Linux, call python <main.py> <forcing.txt>
'''

from fenics import *
from dolfin import *

from soil_params import get_soil_params
from solver import iterated_solver
from aot import aot_solver, get_aot_params
from kalman import kalman_solver
from nudging import nudging_solver


# Params =========================================================

# Set min_x to (negative) depth of soil column in cm. Set nx to the resolution (Default: 2 meter column, with 1 cm spatial resolution)
min_x = -200
max_x = 0
nx = 200
delta_x = (max_x-min_x)/nx

dt = 1                              # Initial/default numerical timestep for Richards solver
aot_dt = 1                          # Initial/default numerical timestep for DA schemes
T = 1440*1                          # Simulation end time (in min)
dt_min = 0.5                        # Minimum numerical timestep
dt_max = 60                         # Maximum numerical timestep (only used if adaptive time stepping is turned on in iterated_solver(), aot_solver(), kalman_solver(), or nudging_solver())

params = get_soil_params()

# Tolerances and max number of iterations for Picard
tol = 1
max_iter = 50

tol_dA = 1
max_iter_dA = 50


# Initialization ===========================================================

mesh = IntervalMesh(nx, min_x, max_x)
V = FunctionSpace(mesh, 'P', 1)

h_0 = Expression('-336.506', degree=0)                  # Actual initial condition
h_0_dA = Expression('-136.506', degree=0)               # Approximate initial condition used in DA schemes


# Retrieve numerical solution to Richards equation
model_time, h_arr, theta_arr = iterated_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0, tol, max_iter)

# Find AOT approximate solution, with observation frequency (in min) given by collect_time and number of observation nodes given by num_nodes.
crns_time = 60          # Pass '0' to turn off; 60 min is typical
sat_time = 0            # Pass '0' to turn off; 1440 min is typical

coll_t = 0
while coll_t < 1:       # Change loop to match arrays in get_aot_params() in aot.py (default is [60, 30, 5, 1] min between observations)
    coll_n = 0
    while coll_n < 1:   # Change loop to match arrays in get_aot_params() in aot.py (default is [5, 7, 10, 12] observations nodes)
        mu, collect_time, num_nodes = get_aot_params(dt, coll_t, coll_n)
        aot_solver(mesh, V, params, min_x, max_x, nx, delta_x, aot_dt, T, dt_min, dt_max, h_0_dA, tol_dA, max_iter_dA, model_time, h_arr, theta_arr, collect_time, num_nodes, crns_time, sat_time)
        coll_n = coll_n + 1
    coll_t = coll_t + 1

# Find nudging approximate solution, with observation frequency (in min) given by collect_time and number of observation nodes given by num_nodes.
coll_t = 0
while coll_t < 1:       # Frequency and spatial values match those from AOT
    coll_n = 0
    while coll_n < 1:
        mu, collect_time, num_nodes = get_aot_params(dt, coll_t, coll_n)
        nudging_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0_dA, 1, 50, model_time, h_arr, theta_arr, collect_time, num_nodes)
        coll_n = coll_n + 1
    coll_t = coll_t + 1

# Find Kalman approximate solution, with observation frequency (in min) given by collect_time and number of observation nodes given by num_nodes.
num_ensembles = 200
coll_t = 0
while coll_t < 1:       # Frequency and spatial values match those from AOT
    coll_n = 0
    while coll_n < 1:
        mu, collect_time, num_nodes = get_aot_params(dt, coll_t, coll_n)
        kalman_solver(mesh, V, params, min_x, max_x, nx, delta_x, dt, T, dt_min, dt_max, h_0_dA, 1, 50, model_time, h_arr, theta_arr, collect_time, num_nodes, num_ensembles)
        coll_n = coll_n + 1
    coll_t = coll_t + 1
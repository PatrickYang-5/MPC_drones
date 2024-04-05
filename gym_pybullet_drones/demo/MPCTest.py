"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `VelocityAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import sys

sys.path.append("../../")
import pybullet as p
import matplotlib.pyplot as plt
# import pytransform3d.visualizer as pv

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
# from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.MPCPIDControl import MPCPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.PathPlanning.GlobalMap import GlobalMap, global_all
from gym_pybullet_drones.PathPlanning.GlobalPathPlanning import GlobalPathPlanning
from gym_pybullet_drones.PathPlanning.MPC import UAV_dynamics, MPC
from gym_pybullet_drones.utils.Drawer import Drawer
from gym_pybullet_drones.MPC.MPCSimple import MPCsimple
from gym_pybullet_drones.MPC.LMPCHover import LMPC, Whole_UAV_dynamics

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = None
# DEFAULT_PHYSICS = Physics("pyb_gnd_drag_dw")
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Global path planning methods
GLOBAL_PLANNER_METHOD = "EasyAStar"  # "AStar" or "RRT"


############ Start the simulation #############################
def run(
        drone=DEFAULT_DRONES,  # Drone model (string)
        num_drones=DEFAULT_NUM_DRONES,  # Number of drones
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,  # Whether to use PyBullet GUI
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3

    # Initializations of the drones(positions, orientations, and goals)

    # Initializations for 2 drones (basic test)
     
    INIT_XYZS = np.array([[0.5,0.5,0.2]])
    INIT_RPYS = np.array([[0, 0, 0]])
    GOAL = np.array([[0.5,1,2]])
    # print("GOAL.shape:", GOAL.shape)
    # print("zeros.shape:", np.zeros(9).shape)

    # state_target = np.hstack([GOAL[0], np.zeros(9), 9.8])
    # print("state_target:", state_target)
    

    # # Initializations for 4 drones
    # INIT_XYZS = np.array([[0.5, 0.5, 0.2], [0.5, 0.5, 1.2], [0.5, 0.2, 0.6], [0.2, 0.5, 0.6], [2.6, 1, 0.5]])
    # INIT_RPYS = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # GOAL = np.array([[3, 1, 1.8], [1, 3, 1.2], [1.5, 2.2, 0.6], [2.2, 1.5, 0.6], [1.7, 1, 0.5]])

    # # Extract two initial states with path conflicts
    # INIT_XYZS = INIT_XYZS[2:5, :]
    # INIT_RPYS = INIT_RPYS[2:5, :]
    # GOAL = GOAL[2:5, :]

    # Establish the global map
    global global_all
    Map = GlobalMap(Length=0.1, range_x=10, range_y=10, range_z=10)  # Set length and range

    #### Create the environment ################################

    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui,
                     global_params=global_all  # Pass the global map to the environment
                     )
    drone_dict = {}
    drone_dict['l'] = env.L
    drone_dict['m'] = env.M
    drone_dict['kf'] = env.KF
    drone_dict['km'] = env.KM
    drone_dict['I'] = env.J
    drone_dict['I_inv'] = env.J_INV
    drone_dict['g'] = env.G
    drone_dict['dt'] = env.CTRL_TIMESTEP
    drone_dict['max_thrust'] = env.MAX_THRUST/4
    # print("l:", l)
    # print("m:", m)
    # print("kf:", kf)
    # print("km:", km)
    #### Initialization and start the planner ################################
    GlobalPlanner = GlobalPathPlanning(method=GLOBAL_PLANNER_METHOD, start=INIT_XYZS.tolist(), goal=GOAL, t=0.1,
                                       num_drones=num_drones)
    # The path is a list of positions without the consideration of the time
    path = GlobalPlanner.Planner()

    obstacle_dic = env.obstacle_dic

    # drawer = Drawer()
    # fig = pv.figure()

    # # draw obstacles and path
    # fig = drawer.draw_obstacle(fig, obstacle_dic)
    # fig = drawer.draw_path(fig, path, line_width=3)

    # fig.show()

    #### Start process the path ###################################
    # repeat the path waypoints to slow down the path
    repeat_count = 10
    path_slow = [[] for i in range(num_drones)]
    for j in range(num_drones):
        path_slow[j] = [element for element in path[j] for i in range(repeat_count)]  # Repeat

    #### Arrange the trajectory ######################
    PERIOD = 50  # Simulation time period
    NUM_WP = control_freq_hz*PERIOD  # Total number of waypoints
    TARGET_POS = np.zeros((NUM_WP, 3, num_drones))  # Target position

    # Get the maximum time step to reach the target
    T_max = np.zeros(num_drones, dtype=int)
    for i in range(num_drones):
        T_max[i] = len(path_slow[i])

    # Start to reach the target
    for j in range(num_drones):
        for i in range(T_max[j]):
            TARGET_POS[i, :, j] = np.array(path_slow[j])[i]

    # Reach the target to end of the simulation
    for j in range(num_drones):
        for i in range(T_max[j], NUM_WP):
            TARGET_POS[i, :, j] = np.array(path_slow[j])[T_max[j]-1]  # Keep the last position

    #### Initialize the way point counters ######################
    wp_counters = np.zeros(num_drones, dtype=int)

    #### Obtain the PyBullet Client ID from the environment ####
    # PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    # logger = Logger(logging_freq_hz=control_freq_hz,
    #                 num_drones=num_drones,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [MPCPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))  # Initialize the control input
    START = time.time()

    dt = 2  # Time step for dynamic model used in MPC
    MPC_N = 5  # Prediction horizon for MPC
    UAV_MPC_control = UAV_dynamics(dt)  # Initialize the dynamic model
    MPC_control = MPC(UAV_MPC_control, MPC_N)  # Initialize the MPC controller
    MPC_whole = Whole_UAV_dynamics(drone_dict)  # Initialize the dynamic model for the whole UAV
    MPC_control_whole = LMPC(MPC_whole, MPC_N)  # Initialize the MPC controller for the whole UAV

    MAX_STEP = 1000  # Maximum number of steps
    state = np.zeros((num_drones, 12))  # Initialize the state
    state[:, 0:3] = INIT_XYZS
    state[:, 6:9] = INIT_RPYS
    history = np.zeros((MAX_STEP, num_drones, 12))  # Initialize the history of the state
    for i in range(MAX_STEP):
        print("Step:", i)
        for j in range(num_drones):
            # Get the next generated position
            generated_pos = np.hstack([TARGET_POS[wp_counters[j], 0:3, j]])
            # Get the next period positions used in MPC
            generated_pos_period = np.hstack([TARGET_POS[wp_counters[j]:(wp_counters[j]+MPC_N+1), 0:3, j]])
            # print("generated_pos_period.shape:", generated_pos_period.shape)
            # print("generated_pos_period:", generated_pos_period)
            # Get the modified position using MPC
            state_target = np.hstack([generated_pos_period, np.zeros((MPC_N+1,9))])
            print("state_target:", state_target)
            optimized_state, optimized_force = MPC_control_whole.MPC_all_state(state[j,:], state_target)
            print("optimized_force:", optimized_force[:, 0])
            print("optimized_state:", optimized_state[:, :].T)
            print("current state:", state[j])
            state = MPC_whole.get_x_next(state[j,:], optimized_force[:, 0])
            print("next state:", state)
            state = state.reshape(-1, 12)
            history[i, j, :] = state[j]

            wp_counters[j] = wp_counters[j]+1 if wp_counters[j]<(NUM_WP-1) else 0
    
    plt.figure()
    plt.plot(range(MAX_STEP),history[:, 0, 0], 'r')
    plt.plot(range(MAX_STEP),history[:, 0, 1], 'g')
    plt.plot(range(MAX_STEP),history[:, 0, 2], 'b')
    plt.show()

if __name__=="__main__":
    run()
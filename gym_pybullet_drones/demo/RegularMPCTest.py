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
import pytransform3d.visualizer as pv

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
DEFAULT_NUM_DRONES = 2
DEFAULT_PHYSICS = None
# DEFAULT_PHYSICS = Physics("pyb_gnd_drag_dw")
DEFAULT_GUI = True
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
GLOBAL_PLANNER_METHOD = "AStar"  # "AStar" or "RRT"


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
     
    INIT_XYZS = np.array([[0.5,0.5,0.2],[0.3,0.3,0.2]])
    INIT_RPYS = np.array([[0, 0, 0],[0,0,0]])
    GOAL = np.array([[0.7,2,1.3],[1.7,1.7,1.3]])
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
    Map = GlobalMap(Length=0.05, range_x=3, range_y=3, range_z=3)  # Set length and range

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

    obstacle_dic = env.obstacle_dic

    ### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    ### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [MPCPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))  # Initialize the control input
    START = time.time()

    dt = 2  # Time step for dynamic model used in MPC
    MPC_N = 50  # Prediction horizon for MPC
    MPC_whole = Whole_UAV_dynamics(drone_dict)  # Initialize the dynamic model for the whole UAV
    MPC_control_whole = LMPC(MPC_whole, MPC_N)  # Initialize the MPC controller for the whole UAV

    MAX_STEP = 150   # Maximum number of steps
    state = np.zeros((num_drones, 12))  # Initialize the state
    for j in range(num_drones):
        state[j, 0:3] = INIT_XYZS[j]
        state[j, 6:9] = INIT_RPYS[j]
    history = np.zeros((MAX_STEP, num_drones, 12))  # Initialize the history of the state
    GOAL_extended = GOAL
    for i in range(MPC_N):
        GOAL_extended = np.hstack([GOAL_extended, GOAL])
    GOAL_extended = GOAL_extended.reshape((num_drones, MPC_N+1, 3))
    for i in range(MAX_STEP):
        
        # Stap the simulation
        env.step_MPC(state)

        print("Step:", i)
        for j in range(num_drones):
            state_target = np.hstack([GOAL_extended[j,:,:], np.zeros((MPC_N+1,9))])
            print("state_target:", state_target)
            print("current state:", state)
            optimized_state, optimized_force = MPC_control_whole.MPC_all_state(state[j], state_target)
            print("optimized_force:", optimized_force[:, 0])
            print("optimized_state:", optimized_state[:, :].T)
            state_j = MPC_whole.get_x_next(state[j,:], optimized_force[:, 0])
            print("next state:", state_j)
            state[j] = state_j
            history[i, j, :] = state[j]
    
    #### End the simulation ####################################

    fig, axes = plt.subplots(num_drones, 3, figsize=(8, 6))
    for i in range(num_drones):
        axes[i,0].plot(range(MAX_STEP),history[:, i, 0], 'r')
        axes[i,0].plot(range(MAX_STEP),np.full((MAX_STEP,), GOAL[i,0]), 'r--')  # Plot the target position
        axes[i,0].set_ylim(-0.5, 2)
        axes[i,0].set_title('drone'+str(i)+'_x')
        axes[i,1].plot(range(MAX_STEP),history[:, i, 1], 'g')
        axes[i,1].plot(range(MAX_STEP),np.full((MAX_STEP,), GOAL[i,1]), 'g--')
        axes[i,1].set_ylim(-0.5, 2.5)
        axes[i,1].set_title('drone'+str(i)+'_y')
        axes[i,2].plot(range(MAX_STEP),history[:, i, 2], 'b')
        axes[i,2].plot(range(MAX_STEP),np.full((MAX_STEP,), GOAL[i,2]), 'b--')
        axes[i,2].set_ylim(-0.5, 2)
        axes[i,2].set_title('drone'+str(i)+'_z')
    plt.tight_layout()  
    plt.show()

if __name__=="__main__":
    run()
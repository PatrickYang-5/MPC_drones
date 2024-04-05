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
import xml.etree.ElementTree as ET
import pytransform3d.visualizer as pv
from pytransform3d.rotations import matrix_from_euler
from matplotlib import colormaps
import pybullet as p
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv
import time


from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
# from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.PathPlanning.GlobalMap import GlobalMap, global_all
from gym_pybullet_drones.PathPlanning.GlobalPathPlanning import GlobalPathPlanning
from gym_pybullet_drones.PathPlanning.MPC import UAV_dynamics, MPC
from gym_pybullet_drones.utils.Drawer import Drawer


start_time = time.time()

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")    
# DEFAULT_PHYSICS = Physics("pyb_gnd_drag_dw")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 50
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# Global path planning methods
GLOBAL_PLANNER_METHOD = "EasyAStar" # "EasyAStar", "AStar" or "RRT"

############ Start the simulation #############################
def run(
        drone=DEFAULT_DRONES,                           # Drone model (string)
        num_drones=DEFAULT_NUM_DRONES,                  # Number of drones
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,                                # Whether to use PyBullet GUI
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
    
    INIT_XYZS = np.array([[3.5, 2, 0.2]])
    INIT_RPYS = np.array([[0, 0, 0]])
    GOAL = np.array([[7,7,1.8]])
    
    '''
    # Initializations for 4 drones
    INIT_XYZS = np.array([[0.5,0.5,0.2],[0.5,0.5,1.2],[0.5,0.2,0.6],[0.2,0.5,0.6]])
    INIT_RPYS = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    GOAL = np.array([[3,1,1.8],[1,3,1.2],[1.5,2.2,0.6],[2.2,1.5,0.6]])

    # Extract two initial states with path conflicts
    INIT_XYZS = INIT_XYZS[2:4,:]
    INIT_RPYS = INIT_RPYS[2:4,:]
    GOAL = GOAL[2:4,:]
    '''

    # Establish the global map
    global global_all
    Map = GlobalMap(Length=0.1, range_x=10, range_y=10, range_z=10) # Set length and range


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
                        global_params=global_all        # Pass the global map to the environment
                        )
    
    #### Initialization and start the planner ################################
    GlobalPlanner = GlobalPathPlanning(method = GLOBAL_PLANNER_METHOD, start=INIT_XYZS.tolist(), goal=GOAL, t=0.1, num_drones=num_drones)
    # The path is a list of positions without the consideration of the time
    path = GlobalPlanner.Planner()
    end_time = time.time()

    elapsed_time = end_time-start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    length_path = 0
    print(path[0][1])
    print(path[0][2])
    path_uav = path[0]
    for i in range(len(path_uav)-1):
        distance = np.sqrt((path[0][i+1][0] - path[0][i][0])**2 + (path[0][i+1][1] - path[0][i][1])**2 + (path[0][i+1][2]- path[0][i][2])**2)
        print(distance)
        length_path = length_path + distance
    print("the length of the path is",length_path)

    obstacle_dic = env.obstacle_dic
    drawer = Drawer()
    fig = pv.figure()

    # draw obstacles and path
    fig = drawer.draw_obstacle(fig, obstacle_dic)
    fig = drawer.draw_path(fig, path, line_width=3)

    fig.show()
    #### Start process the path ###################################
    # repeat the path waypoints to slow down the path
    repeat_count = 10
    path_slow = [[] for i in range(num_drones)]
    for j in range(num_drones):
        path_slow[j] = [element for element in path[j] for i in range(repeat_count)]    # Repeat

    #### Arrange the trajectory ######################
    PERIOD = 20                                                     # Simulation time period
    NUM_WP = control_freq_hz*PERIOD                                 # Total number of waypoints
    TARGET_POS = np.zeros((NUM_WP,3,num_drones))                    # Target position

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
        for i in range(T_max[j],NUM_WP):
            TARGET_POS[i, :, j] = np.array(path_slow[j])[T_max[j]-1] # Keep the last position

    #### Initialize the way point counters ######################
    wp_counters = np.zeros(num_drones, dtype=int)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    record_action = np.zeros((num_drones,4))
    action = np.zeros((num_drones,4))                   # Initialize the control input
    START = time.time()

    '''
    dt = 1                                              # Time step for dynamic model used in MPC
    MPC_N = 5                                           # Prediction horizon for MPC                          
    UAV_MPC_control = UAV_dynamics(dt)                  # Initialize the dynamic model
    MPC_control = MPC(UAV_MPC_control, MPC_N)           # Initialize the MPC controller
    '''
    # Start the simulation loop
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        # Update the env according to the action and get the observation
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract the positions and states from the observation
        positions = np.array([obs[j][0:3] for j in range(num_drones)])
        state = np.array([obs[j][0:6] for j in range(num_drones)])

        #### Compute control for the current way point #############
        # simulate each drone
        for j in range(num_drones):
            # Get the next generated position
            # generated_pos = np.hstack([TARGET_POS[wp_counters[j], 0:3,j]])
            # Get the next period positions used in MPC
            # generated_pos_period = np.hstack([TARGET_POS[wp_counters[j]:(wp_counters[j]+MPC_N), 0:3,j]])
            # Get the modified position using MPC
            # modified_pos, all_pos = MPC_control.MPC_pos(state[j], generated_pos_period, positions[j][:3], positions)

            #### Compute control input using PID and artificial potential field ###############################
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    all_position = positions,
                                                                    current_index = j,
                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:3,j]]),
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )
            print("the action is ",action[j, :])
        record_action = record_action + abs(action)

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2,j], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    print("the total action is ",record_action)
    record_action = record_action[0, :]
    total_action = 0
    for i in record_action:
        total_action = total_action+i
    print(total_action)
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

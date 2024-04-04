import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import cvxpy as cp
import control
import gym_pybullet_drones.MPC.TerminalSet

from gym_pybullet_drones.PathPlanning.GlobalMap import global_all

# UVA dynamics class using simplified model
class Whole_UAV_dynamics():
    ''' The class to describe the dynamics of the UAV'''
    def __init__(self, drone_dict):
        '''
        Parameters:
        ----------------
        dt: the time interval of the drone
        '''
        self.dt = drone_dict['dt']
        self.l = drone_dict['l']
        self.m = drone_dict['m']
        self.g = drone_dict['g']
        self.max_thrust = drone_dict['max_thrust']
        self.I = drone_dict['I']
        self.I_inv = drone_dict['I_inv']
        self.k_f = drone_dict['k_f']
        self.k_m = drone_dict['k_m']
        self.gama = self.k_f/self.k_m
        

        # R_z = np.array([[np.cos(state[8]), -np.sin(state[8]), 0],
        #                 [np.sin(state[8]), np.cos(state[8]), 0],
        #                 [0, 0, 1]])
        # R_y = np.array([[np.cos(state[7]), 0, np.sin(state[7])],
        #                 [0, 1, 0],
        #                 [-np.sin(state[7]), 0, np.cos(state[7])]])
        # R_x = np.array([[1, 0, 0],
        #                 [0, np.cos(state[6]), -np.sin(state[6])],
        #                 [0, np.sin(state[6]), np.cos(state[6])]])
        # R_zyx = R_z @ R_y @ R_x

        # self.A_c = np.zeros((13,13))
        # self.A_c[0:3,3:6] = np.eye(3)
        # self.A_c[6:9,9:12] = np.eye(3)
        # self.A_c[2,12] = -1
        # self.A_c[12,12] = 1
        # # print("self.A_c:",self.A_c)

        
        self.B_c = np.zeros((13,4))
        # self.B_c[3:6,0:3] = R_zyx @ ([0,0,1].T)/m
        self.B_c[9:12,0:3] =np.array([[0,self.l,0, -self.l],
                                        [-self.l,0,self.l, 0],
                                        [-self.gama,self.gama,-self.gama,self.gama]])   
        self.B_c[9:12,0:3] = np.dot(self.I_inv, self.B_c[9:12,0:3])
        # print("self.B_c:",self.B_c)    

        self.C_c = np.eye(13)

        self.D_c = np.zeros((13,4))
        
        # Get the A, B, C, D matrix of the discrete system
        self.A = np.eye(6) + self.A_c * self.dt
        self.B = self.B_c * self.dt
        self.C = self.C_c
        self.D = self.D_c

        # Set the state constraints of the system
        self.x_min = np.array([-20., -20., -20., -0.8, -0.8, -0.8, -np.pi*10/180, -np.pi*10/180, -np.pi*10/180, -20., -20., -20., -10.])
        self.x_max = np.array([-20., -20., -20., -0.8, -0.8, -0.8, -np.pi*10/180, -np.pi*10/180, -np.pi*10/180, -20., -20., -20., -10.])
        # Set the input constraints of the system
        self.u_min = np.array([0., 0., 0., 0.])
        self.u_max = np.array([1., 1., 1., 1.])*self.max_thrust

    def get_x_next(self, x, u):    
        '''
        Parameters:
        ----------------
        self: the class itself
        x: the current state of the drone
        u: the input of the drone
        '''
        return self.A.dot(x) + self.B.dot(u)
    
    def update_matrix(self, state):
        '''
        Parameters:
        ----------------
        self: the class itself
        x: the current state of the drone
        u: the input of the drone
        '''
        R_z = np.array([[np.cos(state[8]), -np.sin(state[8]), 0],
                        [np.sin(state[8]), np.cos(state[8]), 0],
                        [0, 0, 1]])
        R_y = np.array([[np.cos(state[7]), 0, np.sin(state[7])],
                        [0, 1, 0],
                        [-np.sin(state[7]), 0, np.cos(state[7])]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(state[6]), -np.sin(state[6])],
                        [0, np.sin(state[6]), np.cos(state[6])]])
        R_zyx = R_z @ R_y @ R_x

        self.B_c[3:6,0:3] = R_zyx @ ([0,0,1].T)/self.m

        return self.get_x_next(x, u)
    
# MPC class for MPC control
class LMPC():
    ''' The class to make MPC control '''
    def __init__(self, UAV, N):
        '''
        Parameters:
        ----------------
        UAV: the UAV dynamics
        N: the prediction horizon
        '''
        self.UAV = UAV
        self.N = N

    def get_terminal_set(self, A, B, Q, R):
        P,_,G = control.dare(A, B, Q, R)
        return P, G

    def mpc_control(self, x_init, x_target):
        '''
        Parameters:
        ----------------
        self: the class itself
        x_init: the initial state of the drone
        x_target: the target state of the drone
        position_uav: the position of the drone
        position_obs: the position of the obstacles
        '''

        cost = 0.0                                      # The cost function
        constraints = []                                # The constraints                   

        #### Initialize the variables ##############################################
        X = cp.Variable((13, self.N+1))
        u = cp.Variable((4, self.N))
        Q = np.eye(13)
        R = np.eye(4)
        P, K = self.get_terminal_set(self.UAV.A, self.UAV.B, Q, R)

        self.Ak = self.UAV.A - self.UAV.B @ K

        #### Set the constraints and costs ##########################################
        for k in range(self.N):
            x_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.UAV.g])
            cost += cp.quad_form(X[:,k] - x_target, Q)
            u_ref = np.array([1,1,1,1])
            u_ref = u_ref * self.UAV.m * self.UAV.g / 4
            cost += cp.quad_form(u[:,k] - u_ref, R)

            if k == self.N:
                cost += cp.quad_form(X[:,k] - x_target, P)
                constraints += [self.Xf_nr[0] @ (X[:, self.N]-x_target) <= self.Xf_nr[1].squeeze()]

            # Model constraint
            constraints += [X[:,k+1] == self.UAV.A*X[:,k] + self.UAV.B*u[:,k]]
            # State and input constraints
            constraints += [self.UAV.x_min <= X[:,k], X[:,k] <= self.UAV.x_max]
            constraints += [self.UAV.u_min <= u[:,k], u[:,k] <= self.UAV.u_max]


        # Initial constraints
        constraints += [X[:,0] == x_init]

        # Solve the optimization problem using osqp solver
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver = cp.OSQP)

        return X[:,3].value, X[:,:].value


    def constraints_add(self, position_uav, position_obs):
        '''
        Parameters:
        ----------------
        self: the class itself
        position_uav: the position of the drone
        position_obs: the position of the obstacles
        
        Return:
        ----------------
        A,B,C,D: the constraints (planes) of the obstacles
        '''
        nom_vec = position_uav - position_obs
        # print("uav:", position_uav)
        # print("obs:", position_obs)
        non_vec_unit = nom_vec/np.linalg.norm(nom_vec)

        point_center = position_obs - non_vec_unit*10
        A = nom_vec[0]
        B = nom_vec[1]
        C = nom_vec[2]
        D = -A*point_center[0] - B*point_center[1] - C*point_center[2]
        # print(A, B, C, D)
        if ((A*position_uav[0] + B*position_uav[1] + C*position_uav[2] + D) > 0):
            A = -A
            B = -B
            C = -C
            D = -D

        return A,B,C,D
    
    def MPC_all_state(self, state_init, state_target):
        '''
        Parameters:
        ----------------
        self: the class itself
        x_init: the initial state of the drone
        x_target: the target state of the drone
        position_uav: the position of the drone
        position_obs: the position of the obstacles

        Return:
        ----------------
        F: the optimal force of the four rotors
        '''

        F = self.mpc_control(state_init, state_target)
        return F
  
import warnings

# Define a custom filter that filters out
def custom_warning_filter(message, category, filename, lineno, file=None, line=None):
    if "This use of ``*`` has resulted in matrix multiplication" in str(message):
        # ignore the warning
        return None
    # return warnings.defaultaction(message, category, filename, lineno, file)

# Set the custom filter to the warnings module
warnings.showwarning = custom_warning_filter



#### Test the MPC class ########################################################
# dt = 0.1
# T = 10
# X_initial = np.array([0., 0., 0., 0., 0., 0.])
# X_target = np.array([5., 5., 5., 0., 0., 0.])

# UAV_test = UAV_dynamics(dt)

# # for i in range(int(T/dt)):
    
# #     _, X_next, X_current, _ = dummy_control(UAV_test, X_initial, X_target)
# #     X_initial = X_next

# #     print(X_current)

# MPC_test = MPC(UAV_test, 10)

# for i in range(int(T/dt)):
#         A_cc = np.array([
#             [1., 0., 0., 0., 0., 0.],
#             [0., 1., 0., 0., 0., 0.],
#             [0., 0., 1., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0.],
#             [0., 0., 0., 0., 0., 0.]])

#         b_cc = np.array([1., 3., 3., 0., 0., 0.])
#         U, X_next, X_current, _ = MPC_test.mpc_control(X_initial, X_target, A_cc, b_cc)
#         X_initial = X_next
#         print("U:", U)
#         # print("X:",X_next)
#         print("position:",X_current)
#         print("i:",i)
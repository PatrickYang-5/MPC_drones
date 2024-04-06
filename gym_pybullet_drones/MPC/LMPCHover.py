import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import cvxpy as cp
import control
from gym_pybullet_drones.MPC.TerminalSet import Terminal_Set

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
        # print("I:",self.I)
        self.I_inv = drone_dict['I_inv']
        # print("I:",self.I_inv)
        self.k_f = drone_dict['kf']
        self.k_m = drone_dict['km']
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

        self.A_c = np.zeros((12,12))
        self.A_c[0:3,3:6] = np.eye(3)
        self.A_c[6:9,9:12] = np.eye(3)
        self.A_c[3,7] = self.g
        self.A_c[4,6] = -self.g
        # self.A_c[5,12] = -1
        # self.A_c[12,12] = 0
        # print("self.A_c:",self.A_c)

        
        self.B_c = np.zeros((12,4))
        # self.B_c[3:6,0:3] = R_zyx @ ([0,0,1].T)/m
        self.B_c[9:12,0:4] =np.array([[0,self.l,0, -self.l],
                                        [-self.l,0,self.l, 0],
                                        [-self.gama,self.gama,-self.gama,self.gama]])   
        self.B_c[9:12,0:4] = np.dot(self.I_inv, self.B_c[9:12,0:4])
        # print("self.B_c:",self.B_c)    
        self.B_c[5,:] = 1/self.m

        self.A_c = np.array([[0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, self.g, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, (-self.g), 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ])
        
        self.B_c = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1/self.m, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, self.I_inv[0,0], 0, 0],
                       [0, 0, self.I_inv[1,1], 0],
                       [0, 0, 0, self.I_inv[2,2]]])

        self.C_c = np.eye(12)

        self.D_c = np.zeros((12,4))
        
        # Get the A, B, C, D matrix of the discrete system
        self.A = np.eye(12)*1.0 + self.A_c * self.dt
        self.B = self.B_c * self.dt
        self.C = self.C_c
        self.D = self.D_c

        # self.Ad = np.eye(12) + Ac * self.dt
        # self.Bd = Bc * self.dt

        # Set the state constraints of the system
        self.x_min = np.array([-0., -0., -0., -0.8, -0.8, -0.8, -np.pi*10/180, -np.pi*10/180, -np.pi*360/180, -20., -20., -20.])
        self.x_max = np.array([20., 20., 20., 0.8, 0.8, 0.8, np.pi*10/180, np.pi*10/180, np.pi*360/180, 20., 20., 20.])
        # Set the input constraints of the system
        self.u_min = np.array([0., 0., 0., 0.])
        self.u_max = np.array([1., 1., 1., 1.])*self.max_thrust
        self.u_min = -self.u_max

        # Set the terminal constraints of the system
        self.Hx = np.vstack([np.eye(12), -np.eye(12)])
        # self.Hx[12:24, 12:24] = -np.eye(12)
        self.hx = np.vstack([self.x_max.reshape(-1,1), -self.x_min.reshape(-1,1)])
        self.Hu = np.vstack([np.eye(4), -np.eye(4)])
        # self.Hu[4:8, 4:8] = -np.eye(4)
        self.hu = np.vstack([self.u_max.reshape(-1,1), -self.u_min.reshape(-1,1)])
        self.h = np.vstack([self.hu.reshape(-1,1), self.hx.reshape(-1,1)])
        # print("self.Hx:",self.Hx)
        # print("self.hx:",self.hx)
        # print("self.Hu:",self.Hu)
        # print("self.hu:",self.hu)
        # print("self.h:",self.h)

        self.Hx = np.array([
                            [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],  # roll pitch constraints
                            [0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #velocity constraints
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                            [0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0],
                            ])
        
        self.Hu = np.array([[1/self.m, 0, 0, 0],
                            [-1/self.m, 0, 0, 0]
                            ])  # z acc constraints
        
        self.Hu1 = np.array([[1/self.m, 0, 0, 0],
                            [-1/self.m, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, -1.0, 0, 0],
                            [0, 0, 1.0, 0],
                            [0, 0, -1.0, 0],
                            [0, 0, 0, 1.0],
                            [0, 0, 0, -1.0]
                            ])

        self.h = np.array([[0.5*self.g],  # z acc constraints
                           [0.5*self.g],
                           [0.5],  
                           [0.5],
                           [2.0],#velocity constraints
                           [2.0],
                           [2.0],
                           [0.5], 
                           [0.5],
                           [2.0],
                           [2.0],
                           [2.0],
                           [0.5],  
                           [0.5]
                           ])
        self.h1 = np.array([[1.5*self.g],  # z acc constraints
                           [-0.5*self.g],
                           [0.15],  
                           [0.15],
                           [0.15],
                           [0.15],  
                           [0.15],  
                           [0.15],
                           [0.5],  
                           [0.5],
                           [2.0],#velocity constraints
                           [2.0],
                           [2.0],
                           [0.5], 
                           [0.5],
                           [2.0],
                           [2.0],
                           [2.0],
                           [0.5],  
                           [0.5]
                           ])

    def get_x_next(self, x, u):    
        '''
        Parameters:
        ----------------
        self: the class itself
        x: the current state of the drone
        u: the input of the drone
        '''
        G = np.zeros((12))
        G[5] = -self.g*self.dt
        return self.A@x + self.B@u + G
    
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
        #### Initialize the variables ##############################################
        self.X = cp.Variable((12, self.N+1))
        self.u = cp.Variable((4, self.N))
        self.DIR = cp.Parameter((1,self.N))
        self.Q = np.eye(13)*2
        self.R = np.eye(4)*2
        self.Q = np.diag([150, 150, 150, 8, 8, 8, 8, 8, 8, 8, 8, 8])
        self.R = np.diag([10, 10, 10, 10])
        self.E = np.diag([10])
        self.Con_A, self.Con_b, self.Con_A_ext, self.Con_b_ext, self.P = self.get_terminal_set(self.UAV.A, self.UAV.B, self.Q, self.R)

    def get_terminal_set(self, A, B, Q, R):
        # print("A:",A)
        # print("B:",B)
        # print("A_eig:",np.linalg.eig(A)[0])
        P,_,K = control.dare(A, B, Q, R)
        self.Ak = A - B @ K
        #Initial the terminal set object
        # print("Hx:",self.UAV.Hx)
        # print("Hu:",self.UAV.Hu)
        # print("h:",self.UAV.h)
        TerminalSet = Terminal_Set(self.UAV.Hx, self.UAV.Hu, K, self.Ak, self.UAV.h)
        Con_A, Con_b = TerminalSet.Xf
        Con_A_ext, Con_b_ext = TerminalSet.Xf_polygone
        TerminalSet.test(0.15)
        return Con_A, Con_b, Con_A_ext, Con_b_ext, P

    def mpc_control(self, xs, x_target, drone_id):
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
        x_init = xs[drone_id]                           # The initial state of the drone              
        drones_num = xs.shape[0]                        # The number of drones
        #### Set the constraints and costs ##########################################
        for k in range(self.N+1):
            # print("X.shape:",self.X[:,k].shape)
            # print("x_target.shape:",x_target.shape)
            # print("Q.shape:",self.Q.shape)
            if k == self.N:
                cost += cp.quad_form(self.X[:,k] - x_target[k,:], self.P)
                constraints += [self.Con_A_ext @ (self.X[:, self.N]-x_target[k,:]) <= self.Con_b_ext.squeeze()]
                break
            for i in range(drones_num):
                if i < drone_id:
                        o_ini = x_init[0:3]-xs[i][0:3]
                        o_ini_unit = o_ini/np.linalg.norm(o_ini)
                        distance_from_o = 0.2
                        point_on_plane = xs[i][0:3]+distance_from_o*o_ini_unit
                        b = o_ini_unit@point_on_plane
                        constraints += [o_ini_unit@self.X[0:3,k]>=b-self.DIR[0,k]]
                        cost += cp.quad_form(self.DIR[0,k], self.E)
                        # cost += cp.quad_form(1/(self.X[0:3,k]-xs[i][0:3]), self.E)

            cost += cp.quad_form(self.X[:,k] - x_target[k,:], self.Q)
            u_ref = np.array([self.UAV.m*self.UAV.g/4, 0, 0, 0])
            # u_ref = u_ref * self.UAV.m * self.UAV.g / 4
            cost += cp.quad_form(self.u[:,k] - u_ref, self.R)

            

            # Model constraint
            G = np.zeros((12))
            G[5] = -self.UAV.g*self.UAV.dt

            constraints += [self.X[:,k+1] == self.UAV.A@self.X[:,k] + self.UAV.B@self.u[:,k] + G]
            # State and input constraints
            constraints += [self.UAV.Hx @ self.X[:, k] <= self.UAV.h1[self.UAV.Hu1.shape[0]:].squeeze()]
            constraints += [self.UAV.Hu1 @ self.u[:, k] <= self.UAV.h1[:self.UAV.Hu1.shape[0]].squeeze()]


        # Initial constraints
        constraints += [self.X[:,0] == x_init]


        # Solve the optimization problem using osqp solver
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver = cp.OSQP, verbose = False)

        return self.X.value, self.u.value


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
    
    def MPC_all_state(self, states, state_target, drone_id):
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

        X, u = self.mpc_control(states, state_target,drone_id)

        G = np.zeros((12))
        G[5] = -self.UAV.g*self.UAV.dt
        # print(self.UAV.A.shape)
        # print(state_init.shape)
        print("MPC internal", self.UAV.A@states[drone_id].reshape(-1,1) + self.UAV.B@u[:,0] + G)
        print("get x next function", self.UAV.get_x_next(states[drone_id], u[:,0]))
        return X, u
  
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
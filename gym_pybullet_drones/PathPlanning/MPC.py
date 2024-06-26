import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import cvxpy as cp

from gym_pybullet_drones.PathPlanning.GlobalMap import global_all

# UVA dynamics class using simplified model
class UAV_dynamics():
    ''' The class to describe the dynamics of the UAV'''
    def __init__(self, dt):
        '''
        Parameters:
        ----------------
        dt: the time interval of the drone
        '''

        self.A_c = np.array([
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        
        self.B_c = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

        self.C_c = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.]])

        self.D_c = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
        
        # Get the A, B, C, D matrix of the discrete system
        self.A = np.eye(6) + self.A_c * dt
        self.B = self.B_c * dt
        self.C = self.C_c
        self.D = self.D_c

        # Set the state constraints of the system
        self.x_min = np.array([-100., -100., -100., -30., -30., -30.])
        self.x_max = np.array([100., 100., 100., 30., 30., 30.])
        # Set the input constraints of the system
        self.u_min = np.array([-20., -20., -20.])
        self.u_max = np.array([20., 20., 20.])

    def get_x_next(self, x, u):    
        '''
        Parameters:
        ----------------
        self: the class itself
        x: the current state of the drone
        u: the input of the drone
        '''
        return self.A.dot(x) + self.B.dot(u)
    
# MPC class for MPC control
class MPC():
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


    def mpc_control(self, x_init, x_target, position_uav, position_obs):
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
        X = cp.Variable((6, self.N+1))
        u = cp.Variable((3, self.N))
        x_another_obver = np.array([1.6,1,0.5])
        weight_input = 0.2*np.eye(3)
        weight_tracking = 3.0*np.eye(3)
        o_ini = position_uav-x_another_obver
        o_ini_unit = o_ini/np.linalg.norm(o_ini)
        distance_from_o = 0.1
        point_on_plane = x_another_obver+distance_from_o*o_ini_unit
        b = o_ini_unit@point_on_plane

        #### Set the constraints and costs ##########################################
        for k in range(self.N):
            # Cost function
            cost += (X[0,k]-x_target[k][0])**2*weight_tracking[0,0] + (X[1,k]-x_target[k][1])**2*weight_tracking[1,1] + (X[2,k]-x_target[k][2])**2*weight_tracking[2,2]
            cost += u[0,k]**2*weight_input[0,0] + u[1,k]**2*weight_input[1,1] + u[2,k]**2*weight_input[2,2]

            # Model constraint
            constraints += [X[:,k+1] == self.UAV.A*X[:,k] + self.UAV.B*u[:,k]]
            # State and input constraints
            constraints += [self.UAV.x_min <= X[:,k], X[:,k] <= self.UAV.x_max]
            constraints += [self.UAV.u_min <= u[:,k], u[:,k] <= self.UAV.u_max]

            # Obstacle (other drones) constraints
            constraints += [o_ini_unit@X[0:3,k+1]>=b+0.1]

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
    
    def MPC_pos(self, x_init, x_target, position_uav, position_obs):
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
        x_next: the next state of the drone
        x_all: the all states of the drone in the prediction horizon
        '''

        x_next, x_all = self.mpc_control(x_init, x_target, position_uav, position_obs)
        return x_next, x_all
  
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
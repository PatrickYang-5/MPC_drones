import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class Terminal_Set:
    def __init__(self, H_x, H_u, K, A_k, h):
        self.H_x = H_x
        self.H_u = H_u
        self.K = K
        self.A_k = A_k
        self.h = h
        self.H = np.block([[H_u, np.zeros((H_u.shape[0], H_x.shape[1]))], [np.zeros((H_x.shape[0], H_u.shape[1])), H_x]])
        self.N_c = self.H.shape[0]
        self.N_x = A_k.shape[1]
        self.K_aug = np.vstack([-K, np.eye(self.N_x)])
        self.Iterations = 200
        self.Xf = self.ComputeTerminalSet()
        self.Xf_polygone = self.ComputeTerminalSetPolytope()
        self.test(0.15)

    def ComputeTerminalSet(self):
        # The constraints' slope
        Con_A = np.zeros((0, self.N_x))

        # The constraints' intercept
        Con_b = np.zeros((0, 1))

        self.C = self.H @ self.K_aug
        Ft = np.eye(self.N_x)

        for i in range(self.Iterations):
            # Compute the constraints' slope and intercept
            Con_A = np.vstack([Con_A, self.C @ Ft])
            Con_b = np.vstack([Con_b, self.h])

            # Compute the new terminal cost
            Ft = self.A_k @ Ft
            f_obj = self.C @ Ft
            violation = False

            for j in range(self.N_c):
                val = self.get_value(f_obj[j,:], Con_A, Con_b)
                print(val > self.h[j])
                if val > self.h[j]:
                    violation = True
                    break

            if not violation:
                return [Con_A, Con_b]
            
    def get_value(self, f_obj, A, b):
        # Define the optimization variables
        x = cp.Variable((self.N_x,1))
        objective = cp.Maximize(f_obj @ x)
        constraints = [A @ x <= b]
        prob = cp.Problem(objective, constraints)
        # Define the optimization problem
        # prob = cp.Problem(cp.Maximize(f_obj @ x), [A @ x <= b])

        # Solve the optimization problem
        value = prob.solve(verbose=False)

        return value
    
    def ComputeTerminalSetPolytope(self):
        # print(self.Xf)
        Con_A, Con_b = self.Xf
        Con_A_ext, Con_b_ext = Con_A.copy(), Con_b.copy()
        i = 0
        while i < Con_A_ext.shape[0]:
            obj = Con_A_ext[i,:]
            Con_b_large = Con_b_ext.copy()
            Con_b_large[i] = Con_b_large[i] + 1
            val = self.get_value(obj, Con_A_ext, Con_b_large)
            if val <= Con_b_ext[i]:
                Con_A_ext = np.delete(Con_A_ext, i, axis=0)
                Con_b_ext = np.delete(Con_b_ext, i, axis=0)
            else:
                i += 1
        return [Con_A_ext, Con_b_ext]
    
    def test(self,limit = 0.15):
        Con_A, Con_b = self.Xf_polygone
        violation = False
        for i in range(4):
            x = cp.Variable((12,1))
            u = cp.Variable((4,1))
            cost = 0
            constr = []
            constr.append(Con_A @ x[:,0] <= Con_b.squeeze())
            constr.append(u[:,0] == -self.K @ x[:,0])
            cost = u[i,0]
            prob = cp.Problem(cp.Maximize(cost), constr)
            prob.solve()
            print("input",i,'<',prob.value)
            if prob.value > limit:
                violation = True
        if violation == False:
            print("Passed")


            

        
        


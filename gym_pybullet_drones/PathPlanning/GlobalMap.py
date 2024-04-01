import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### global variable ##############################
global_all = {"map": None,              # the map of the workspace
              "length": None,           # the length of the grid
              "range_x": None,          # the range of the workspace
              "range_y": None,
              "range_z": None}

#### GlobalMap class #############################
class GlobalMap():
    ''' The class to store the global map '''

    def __init__(self, Length, range_x, range_y, range_z):
        '''
        Parameters:
        Length: the length of the grid
        range_(x,y,z): the range of the workspace
        Map: the map of the workspace
        global_data: the global data of the hole program
        '''
        # Set the self variables
        self.Length = Length
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z
        self.Map = np.zeros((int(range_x/Length), int(range_y/Length), int(range_z/Length)))
        self.global_data = {"map": self.Map,
                            "length": self.Length,
                            "range_x": self.range_x,
                            "range_y": self.range_y,
                            "range_z": self.range_z}
        
        # Update the global variables
        global global_all
        global_all["map"] = self.global_data["map"]
        global_all["length"] = self.global_data["length"]
        global_all["range_x"] = self.global_data["range_x"]
        global_all["range_y"] = self.global_data["range_y"]
        global_all["range_z"] = self.global_data["range_z"]
        # print("Global Map is initialized.1111111111111111111111111111")


    # Check the map
    def checkMap(self):
        '''
        Check the map
        
        Parameters:
        ----------------
        self: the class itself
        '''

        # Start plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim(-0.5*self.Map.shape[0]*self.Length, 0.5*self.Map.shape[0]*self.Length)
        ax.set_ylim(-0.5*self.Map.shape[0]*self.Length, 0.5*self.Map.shape[0]*self.Length)
        ax.set_zlim(0, self.Map.shape[2]*self.Length)

        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                for k in range(self.Map.shape[2]):
                    if self.Map[i][j][k] == 1:
                        ax.scatter(i*self.Length, j*self.Length, k*self.Length, c='b', marker='o')

        plt.show()

        
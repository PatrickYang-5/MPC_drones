import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq

from gym_pybullet_drones.PathPlanning.GlobalMap import global_all

# class to record the information of the node
class Node():
    ''' The class to record the information of the node'''
    def __init__(self, position=[0,0,0], cost=None, heuristic=None, parent=None):
        '''
        Parameters:
        ----------------
        position: the position of the node
        cost: the cost of the node
        heuristic: the heuristic of the node
        parent: the parent of the node
        '''
        self.position = position
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

# AStar class for AStar algorithm
class EasyAStar():
    ''' The class to make AStar algorithm '''
    def __init__(self, start=[0,0,0], goal=[1,1,1], t=0.1, num_drones=1):
        '''
        Parameters:
        ----------------
        start: the start point of the drone
        goal: the goal point of the drone
        t: the time interval of the drone
        num_drones: the number of the drones
        '''
        self.t = t
        self.global_data = global_all
        self.start = [[round(start[i][0]/self.global_data["length"]), \
                     round(start[i][1]/self.global_data["length"]), \
                     round(start[i][2]/self.global_data["length"])]for i in range(num_drones)]
        self.goal = [[round(goal[i][0]/self.global_data["length"]), \
                    round(goal[i][1]/self.global_data["length"]), \
                    round(goal[i][2]/self.global_data["length"])]for i in range(num_drones)] 
        self.num_drones = num_drones


    # define a function to check if the position is valid
    def is_valid_position(self, position,flag):
        '''
        Parameters:
        ----------------
        position: the position of the node
        flag: the number of the drone
        '''
        x, y, z = position

        # return 1 if the position is valid
        return x >= 0 and x < self.global_data["map"].shape[0] \
            and y >= 0 and y < self.global_data["map"].shape[1] \
            and z >= 0 and z < self.global_data["map"].shape[2] \
            and self.global_data["map"][x][y][z] == 0 or self.global_data["map"][x][y][z] == flag+1

    # define a function to get the neighbor from a node
    def get_nei_cos(self, position,flag):
        '''
        Parameters:
        ----------------
        position: the position of the node
        flag: the number of the drone

        Return:
        ----------------
        neighbors: the neighbors of the node
        costs: the costs of the neighbors
        '''
        x, y, z = position
        neighbors = []
        costs = []
        i_s = []
        j_s = []
        k_s = []

        # get the neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    neighbor = [x+i, y+j, z+k]
                    if neighbor != position and self.is_valid_position(neighbor,flag):
                        neighbors.append(neighbor)
                        i_s.append(i)
                        j_s.append(j)
                        k_s.append(k)
                        if abs(i)+abs(j)+abs(k) == 1:   # neighbor is a direct neighbor
                            costs.append(1)
                        elif abs(i)+abs(j)+abs(k) == 2: # neighbor is a 2D-diagonal neighbor
                            costs.append(1.4)
                        else:                           # neighbor is a 3D-diagonal neighbor
                            costs.append(1.7)
        return neighbors, costs, i_s, j_s, k_s

    # define the heuristic parameter
    def heuristic(self, position,i):
        '''
        Return:
        ----------------
        the heuristic parameter
        '''
        return (position[0]-self.goal[i][0])**2 + (position[1]-self.goal[i][1])**2 + (position[2]-self.goal[i][2])**2
    

    def graph_clean(self,flag):
        '''
        Function: Clean the obstacle in the graph in the last step

        Parameters:
        ----------------
        flag: the number of the drone
        '''
        # print("flag+1:", flag+1)
        for i in range(self.global_data["map"].shape[0]):
            for j in range(self.global_data["map"].shape[1]):
                for k in range(self.global_data["map"].shape[2]):
                    if self.global_data["map"][i][j][k] == flag+1:
                        self.global_data["map"][i][j][k] = 0

    def graph_change(self, position,flag):
        '''
        Function: Change the obstacle in the graph in the current step

        Parameters:
        ----------------
        position: the position of the node
        flag: the number of the drone
        '''

        for i in range(int(position[0]-0.2/self.global_data["length"]), int(position[0]+0.2/self.global_data["length"]+1)):
            for j in range(int(position[1]-0.2/self.global_data["length"]), int(position[1]+0.2/self.global_data["length"]+1)):
                for k in range(int(position[2]-0.2/self.global_data["length"]), int(position[2]+0.2/self.global_data["length"]+1)):
                    if self.global_data["map"][i][j][k] == 0:           # if the position is not occupied (avoid the conflict)
                        self.global_data["map"][i][j][k] = flag+1

    # define the AStar algorithm
    def EasyAstar(self):
        '''
        Function: AStar algorithm
        '''
        open_list = [[] for i in range(self.num_drones)]        # the open list that stores the nodes that have not been visited
        closed_list = [[] for i in range(self.num_drones)]      # the closed list that stores the nodes that have been visited
        current_node = [[] for i in range(self.num_drones)]     # the current node
        path = [[] for i in range(self.num_drones)]             # the path for all the drones
        start_node = [[] for i in range(self.num_drones)]       # the start node

        # define the start node
        for i in range(self.num_drones):
            start_node[i] = Node(self.start[i], 0, self.heuristic(self.start[i],i), None)
            open_list[i].append(start_node[i])

        # define the flag
        flag = 0        # the number of the drone
        flag_term = 0   # the terminal flag to judge how many loops have been done

        # init the current node and the closed list
        for i in range(self.num_drones):
            current_node[i] = heapq.heappop(open_list[i])
            closed_list[i].append(current_node[i])

        while any(len(open_list[i]) for i in range(self.num_drones)) > 0 or flag_term == 0:
            # print("which drone is being precessed:", flag)    # print the number of the drone that is being processed

            # only update the current node when it is the first loop
            if flag_term >= self.num_drones:
                current_node[flag] = heapq.heappop(open_list[flag])
                closed_list[flag].append(current_node[flag])

            #### update the graph   ##########################################
            # self.graph_clean(flag)
            # self.graph_change(current_node[flag].position,flag)
            # self.graph_show()

            flag_term += 1

            # show the path when all drones reach their goal
            if all(current_node[j].position == self.goal[j] for j in range(self.num_drones)):

                for j in range(self.num_drones):
                    path[j] = []
                    while current_node[j].parent != None:               # get the path    
                        path[j].append(current_node[j].position)
                        current_node[j] = current_node[j].parent
                    path[j].append(current_node[j].position)
                    path[j].reverse()                                   # reverse the path
                
                # return the processed path that is in the real world
                return [[[element * self.global_data["length"] for element in sub_sub_list] for sub_sub_list in sub_list] for sub_list in path]  
            
            # Jump to the next drone if the drone reaches its goal
            if current_node[flag].position == self.goal[flag]:
                flag = (flag+1)%self.num_drones
                continue
                
            # get the neighbor
            neighbors,costs,i_s,j_s,k_s = self.get_nei_cos(current_node[flag].position,flag)
            for neighbor,cost,i_u,j_u,k_u in zip(neighbors,costs,i_s,j_s,k_s):
                neighbor_node = Node(neighbor, current_node[flag].cost + cost, self.heuristic(neighbor,flag), current_node[flag])
                    
                ## test the cost
                # print("ijk:", i_u, j_u, k_u)
                # print("costs: ", cost)

                if neighbor_node in closed_list[flag]:
                    continue
                if neighbor_node not in open_list[flag]:
                    heapq.heappush(open_list[flag], neighbor_node)
                else:
                    for node in open_list[flag]:
                        if node == neighbor_node and node.cost > neighbor_node.cost:
                            node.cost = neighbor_node.cost
                            node.parent = neighbor_node.parent
            flag+=1
            flag = flag%self.num_drones
        return None

    # show the path
    def checkPath(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.set_xlim(-0.5*self.global_data["map"].shape[0]*self.global_data["length"], 0.5*self.global_data["map"].shape[0]*self.global_data["length"])
        # ax.set_ylim(-0.5*self.global_data["map"].shape[1]*self.global_data["length"], 0.5*self.global_data["map"].shape[1]*self.global_data["length"])
        # ax.set_zlim(0, self.global_data["map"].shape[2]*self.global_data["length"])


        for i in range(self.global_data["map"].shape[0]):
            for j in range(self.global_data["map"].shape[1]):
                for k in range(self.global_data["map"].shape[2]):
                    if self.global_data["map"][i][j][k] == self.num_drones+2:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='b', marker='o')
                        # ax.scatter(1.6,1,0.5)
        for i in range(self.num_drones):
            color = np.linspace(0, 1, len(path[i]))
            path_loc = np.array(path[i])
            path_x = path_loc[:, 0]
            path_y = path_loc[:, 1]
            path_z = path_loc[:, 2]
            ax.scatter(path_x, path_y, path_z, c=color, marker='o', cmap='viridis')

        # for i in range(len(path)):
        #     color += 1/len(path)
        #     ax.scatter(path[i][0], path[i][1], path[i][2], c=0.9, marker='o', cmap='viridis')

        plt.show()

    def graph_show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # ax.set_xlim(-0.5*self.global_data["map"].shape[0]*self.global_data["length"], 0.5*self.global_data["map"].shape[0]*self.global_data["length"])
        # ax.set_ylim(-0.5*self.global_data["map"].shape[1]*self.global_data["length"], 0.5*self.global_data["map"].shape[1]*self.global_data["length"])
        # ax.set_zlim(0, self.global_data["map"].shape[2]*self.global_data["length"])

        for i in range(self.global_data["map"].shape[0]):
            for j in range(self.global_data["map"].shape[1]):
                for k in range(self.global_data["map"].shape[2]):
                    if self.global_data["map"][i][j][k] == 1:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='b', marker='o')
                    if self.global_data["map"][i][j][k] == 2:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='r', marker='o')
                    if self.global_data["map"][i][j][k] == 3:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='g', marker='o')
                    if self.global_data["map"][i][j][k] == 4:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='y', marker='o')
                    # if self.global_data["map"][i][j][k] == 5:
                    #     ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='c', marker='o')
                    if self.global_data["map"][i][j][k] == 6:
                        ax.scatter(i*self.global_data["length"], j*self.global_data["length"], k*self.global_data["length"], c='m', marker='o')

        plt.show()
        

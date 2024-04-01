import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from gym_pybullet_drones.PathPlanning.GlobalMap import global_all

# create a class variable to record the information of the node
class Node():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.parents = None
        self.cost = 0

class RRT_star():
    def __init__(self,start_list,goal,t):
        self.global_data = global_all
        start = np.array(start_list[0])
        self.start = [int(start[0]/self.global_data["length"]), \
                 int(start[1]/self.global_data["length"]), \
                 int(start[2]/self.global_data["length"])]

        goal = goal[0]
        self.goal = [int(goal[0]/self.global_data["length"]), \
                int(goal[1]/self.global_data["length"]), \
                int(goal[2]/self.global_data["length"])]
        start_node = Node(start[0], start[1], start[2])
        self.t = t
        # create the constraint in the map, the coordinate of the constraint is 1
        self.Map = self.global_data["map"]

        x_max = 0.8*self.Map.shape[0]
        y_max = 0.8*self.Map.shape[1]
        z_max = 0.3*self.Map.shape[2]
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
    # define the function to show the RRT process
    def rrt_star(self):
        # create the node variable to save the information of the node
        nodes = []
        Map = self.Map

        detect_range = 30# the range of the node to detect its parents
        end_size = 2 # the range to end the loop
        threshold = 20 # the range to get new points
        steps = 500 # the maximiun step of the loop

        # add start node to the set
        start_node = Node(self.start[0], self.start[1], self.start[2])
        start_node.cost = 0
        nodes.append(start_node)

        # start the searching loop
        for j in range(steps):
            # choose the node from the map
            while True:
                # randomly choose a node from the map
                random_x = np.random.randint(30, self.x_max)
                random_y = np.random.randint(20, self.y_max)
                random_z = np.random.randint(0, self.z_max)
                # check whether this node has been picked before
                check_double = self.check_double_set(nodes,random_x,random_y,random_z)

                # check whether this node cannot be linked to the others
                check_obstacle = self.check_obstacle_others(nodes,random_x,random_y,random_z)

                # check whether this node is close to others
                check_near = self.check_near_others(nodes,random_x,random_y,random_z,threshold)

                # detect whether the node is in the constraint
                if Map[random_x,random_y,random_z]!=1 & check_double & check_obstacle & check_near == True:
                    break
            new_node = Node(random_x,random_y,random_z)

            # get the related node in the range
            related_node,find_in_range = self.find_node_in_range(nodes,new_node,detect_range)

            # if there is not any nodes in the range
            if find_in_range == 0:
                cloest_node = self.find_cloest_node(nodes,new_node)
                new_node.parents = cloest_node
                nodes.append(new_node)
                continue

            # detect the collision in the related nodes and get the parent node
            parents_node = []
            for j in range(len(related_node)):
                if self.detect_collision(new_node,related_node[j])==0:
                    parents_node.append(related_node[j])

            # if all the related nodes are hindered
            if len(parents_node)==0:
                nodes.append(new_node)
                continue

            # get the parent in the parent cost
            new_node.parents,new_node.cost = self.get_parent(parents_node, start_node)

            # renew the parent_node in the range
            for i in range(len(parents_node)):
                cost_i = self.distance(new_node, parents_node[i])+new_node.cost
                if cost_i<parents_node[i].cost:
                    parents_node[i].parents = new_node
                    parents_node[i].cost = cost_i

            nodes.append(new_node)

        # find the cloest node to the goal
        goal_node = Node(self.goal[0],self.goal[1],self.goal[2])
        #goal_node_range = goal_node.find_node_in_range(nodes, new_node, detect_range)
        goal_node_cloest = self.find_cloest_node(nodes,goal_node)
        goal_node.parents = goal_node_cloest
        nodes.append(goal_node)
        self.nodes = nodes
        return nodes

    # check whether this node has been picked before
    def check_double_set(self,nodes,random_x,random_y,random_z):
        check = True
        for node in nodes:
            if node.x == random_x & node.y == random_y & node.z == random_z:
                check = False
        return check

    # check whether this node cannot be linked to the others
    def check_obstacle_others(self,nodes,random_x,random_y,random_z):
        check = False
        Map = self.Map
        for node in nodes:
            new_node = Node(random_x,random_y,random_z)
            if self.detect_collision(node, new_node)==0:
                check = True
        return check

    # check whether this point is closed enough to others
    def check_near_others(self,nodes,random_x,random_y,random_z,threshold):
        check = False
        for node in nodes:
            new_node = Node(random_x,random_y,random_z)
            if self.distance(node, new_node) <= threshold:
                check = True
        return check
    # define a function to get the related node in the range
    def find_node_in_range(self,nodes,node,detect_range):

        related_node = []
        find_in_range = 0# variable to know that if there is a point in the range
        for j in range(len(nodes)):
            dis = self.distance(nodes[j],node)
            if dis<=detect_range:
                related_node.append(nodes[j])
                find_in_range = 1
        return related_node,find_in_range

    # define a function to get the cloest node
    def find_cloest_node(self,nodes,node):
        cloest_num = 0
        Map = self.Map
        cloest_dis = self.distance(nodes[0],node)
        for j in range(len(nodes)):
            if self.detect_collision(node,nodes[j]) == 1:
                continue
            else:
                dis = self.distance(nodes[j],node)
                if dis<=cloest_dis:
                    cloest_num = j
                    cloest_dis = dis
        return nodes[cloest_num]

    # define a function to get the distance between nodes
    def distance(self,node_1,node_2):
        dis_x = node_1.x-node_2.x
        dis_y = node_1.y-node_2.y
        dis_z = node_1.z-node_2.z
        dis = np.sqrt(dis_x**2+dis_y**2+dis_z**2)
        return dis

    # define a function to detect the collision
    def detect_collision(self,node_1,node_2):
        collision_detect = 0
        Map = self.Map
        hundred_x = np.linspace(node_1.x, node_2.x, num=100)
        hundred_y = np.linspace(node_1.y, node_2.y, num=100)
        hundred_z = np.linspace(node_1.z, node_2.z, num=100)
        for i in range(100):
            if Map[round(hundred_x[i]),round(hundred_y[i]),round(hundred_z[i])] != 0:
                collision_detect = 1
                break
        return collision_detect

    # define a function to get the parent node
    def get_parent(self,parent_node,start_node):
        parent_nodes = []
        for pare_node in parent_node:
            parent_nodes.append(pare_node)
        parent_node = parent_nodes

        cost = self.distance(start_node,parent_node[0])+parent_node[0].cost
        num = 0
        distance = self.distance(parent_node[0],start_node)
        for i in range(len(parent_node)-1):
            cost_i = self.distance(parent_node[i+1],start_node)+parent_node[i+1].cost
            if cost_i < cost:
                cost = cost_i
                num = i+1
        return parent_node[num],cost

    def RRT_process(self):
        '''
        # set the start and the end point
        start = np.array([0,0,0])
        goal = np.array([9.9,9.9,9.9])
        '''
        # start the loop

        nodes = self.rrt_star()
        process = []
        process.append(nodes[-1])
        node = nodes[-1].parents
        while node.x != nodes[0].x or node.y != nodes[0].y or node.z != nodes[0].z :
            process.append(node)
            node = node.parents
        process.append(nodes[0])

        process.reverse()
        print("the lenth",len(process))
        path = []
        for every_node in process:
            path.append([every_node.x*self.global_data["length"],every_node.y*self.global_data["length"],every_node.z*self.global_data["length"]])
        path = np.array(path)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_zlim(0, 50)

        for node in nodes:
            if node.x!=nodes[0].x or node.y != nodes[0].y or node.z != nodes[0].z :
                ax.plot(node.x, node.y, node.z, marker='o', color='green')
            else:
                ax.plot(node.x, node.y, node.z, marker='o', color='red')
            if node.parents!=None:
                ax.plot([node.x, node.parents.x], [node.y, node.parents.y], [node.z, node.parents.z], linestyle='-',
                        color='blue')
            # plt.draw()
            # plt.pause(0.5)
        current_parents = nodes[-1].parents
        node_path_1 = nodes[-1]
        node_path_2 = nodes[-1].parents
        i = 0
        while current_parents!=None:
            print(i)
            ax.plot([node_path_1.x, node_path_2.x], [node_path_1.y, node_path_2.y], [node_path_1.z, node_path_2.z],
                    linestyle='-', color='red')
            node_path_1 = node_path_2
            if node_path_1.parents==None:
                break
            node_path_2 = node_path_1.parents
            current_parents = node_path_2.parents
            i += 1
        length = self.global_data["length"]
        for i in range(int(self.x_max)):
            for j in range(int(self.y_max)):
                for k in range(int(self.z_max)):
                    if self.Map[i][j][k]==1:
                        a = ax.scatter(i, j, k,c = 'b')
                        print(i,j,k)
        ax.plot([node_path_2.x, node_path_1.x], [node_path_2.y, node_path_1.y], [node_path_2.z, node_path_1.z],
                linestyle='-', color='red')
        plt.show()

        return path
'''
#for node in nodes:
#    print('x')
#    print(node.x)
#    print('y')
#    print(node.y)
    #print(node.cost)
    #print("this is parent")
    #print(node.parents)


# show the graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.set_zlim(0,100)

for i in range(100):
    for j in range(100):
        for k in range(100):
            if Map[i][j][k]==1:
                ax.plot(i, j, k, marker="o", color='red')

# 定义立方体的顶点坐标
vertices = [
    [75, 75, 75], [25, 75, 75], [25, 25, 75], [75, 25, 75],
    [75, 75, 25], [25, 75, 25], [25, 25, 25], [75, 25, 25]
]

# 定义立方体的面
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],
    [vertices[4], vertices[5], vertices[6], vertices[7]],
    [vertices[0], vertices[1], vertices[5], vertices[4]],
    [vertices[2], vertices[3], vertices[7], vertices[6]],
    [vertices[1], vertices[2], vertices[6], vertices[5]],
    [vertices[4], vertices[7], vertices[3], vertices[0]]
]

# 绘制立方体的各个面
for face in faces:
    x = [point[0] for point in face]
    y = [point[1] for point in face]
    z = [point[2] for point in face]
    ax.plot(x, y, z, color='red')

for node in nodes:
    if node.x!=goal[0] or node.y!=goal[1]:
        ax.plot(node.x,node.y,node.z,marker = 'o',color = 'green')
    else:
        ax.plot(node.x, node.y,node.z, marker='o', color='red')
    if node.parents != None:
        ax.plot([node.x,node.parents.x], [node.y,node.parents.y], [node.z,node.parents.z], linestyle='-', color='blue')
    #plt.draw()
    #plt.pause(0.5)
current_parents = nodes[-1].parents
node_path_1 = nodes[-1]
node_path_2 = nodes[-1].parents
i=0
while current_parents!=None:
    print(i)
    ax.plot([node_path_1.x,node_path_2.x], [node_path_1.y,node_path_2.y], [node_path_1.z,node_path_2.z], linestyle='-', color='red')
    node_path_1 = node_path_2
    if node_path_1.parents == None:
        break
    node_path_2 = node_path_1.parents
    current_parents = node_path_2.parents
    i +=1
ax.plot([node_path_2.x,node_path_1.x], [node_path_2.y,node_path_1.y], [node_path_2.z,node_path_1.z], linestyle='-', color='red')
plt.show()
'''



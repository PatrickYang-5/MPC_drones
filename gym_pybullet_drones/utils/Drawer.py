import xml.etree.ElementTree as ET
import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.rotations import matrix_from_euler
from matplotlib import colormaps

# class with methods to draw obstacles and paths in 3D
class Drawer:

    def __init__(self):
        pass

    def read_urdf_shape(self, urdf_file):
        '''
        Function: Read shape and dimension from the urdf file, only support box, sphere and cylinder

        Parameters:
        ----------------
        urdf_file: path of the urdf file

        Return:
        ----------------
        shape_type: shape type in string
        dimension: dimension of the shape, [radius] for sphere, [radius, length] for cylinder, [x, y, z] for box
        '''
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        shape_type = None
        dimension = None

        # find the geometry tag
        for link in root.findall('.//link'):
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    shape = geometry.find('*')
                    if shape is not None:
                        shape_type = shape.tag

                        # read dimension
                        if shape_type == 'sphere':
                            radius = float(shape.attrib.get('radius', 0))
                            dimension = [radius]

                        elif shape_type == 'box':
                            size = shape.attrib.get('size')
                            if size is not None:
                                dimension = [float(dim) for dim in size.split()]

                        elif shape_type == 'cylinder':
                            radius = float(shape.attrib.get('radius', 0))
                            length = float(shape.attrib.get('length', 0))
                            dimension = [radius, length]

        return shape_type, dimension

    def p_ruler_to_T(self, p, ruler):
        '''
        Function: Convert position and euler angle to transformation matrix

        Parameters:
        ----------------
        p: position
        ruler: euler angles

        Return:
        ----------------
        T: transformation matrix
        '''
        T = np.eye(4)
        T[:3, 3] = p
        T[:3, :3] = matrix_from_euler(ruler, 0, 1, 2, False)
        return T

    def draw_obstacle(self, fig, obstacle_dict):
        '''
        Function: Draw obstacles in the figure

        Parameters:
        ----------------
        fig: figure
        obstacle_dict: dictionary of obstacles, including file_name, baseposition and ruler

        Return:
        ----------------
        fig: figure
        '''
        for i in range(len(obstacle_dict['file_name'])):
            # read shape, dimension and get transformation matrix
            shape_type, dimension = self.read_urdf_shape(obstacle_dict['file_name'][i])
            baseposition = obstacle_dict['baseposition'][i]
            ruler = obstacle_dict['ruler'][i]
            T = self.p_ruler_to_T(baseposition, ruler)
            # color = np.array([150, 150, 150])/255

            # draw obstacles
            if shape_type == 'box':
                fig.plot_box(size=dimension, A2B=T)

            elif shape_type == 'sphere':
                fig.plot_sphere(radius=dimension[0], A2B=T)

            elif shape_type == 'cylinder':
                fig.plot_cylinder(length=dimension[1], radius=dimension[0], A2B=T)

            else:
                print("shape type is not supported: " + shape_type)

        return fig

    def draw_path(self, fig, paths, line_width=3, colormap="viridis"):
        '''
        Function: Draw path in the figure

        Parameters:
        ----------------
        fig: figure
        paths: path, list of list of points (num. of paths * num. of points in a path * 3)
        line_width: width of the line

        Return:
        ----------------
        fig: figure
        '''
        fig.set_line_width(line_width)
        for i in range(len(paths)):
            colors = colormaps[colormap].resampled(len(paths[i]) - 1).colors[:, :3]
            fig.plot(paths[i], c=colors)
        return fig


test = False

if test:
    # test code
    obstacle_dic = {}
    obstacle_dic['file_name'] = ["../assets/small_box.urdf",
                                 "../assets/small_box.urdf",
                                 "../assets/small_box.urdf",
                                 "../assets/small_box.urdf",
                                 "../assets/cuboid.urdf",
                                 "../assets/sphere.urdf"
                                 ]
    obstacle_dic['baseposition'] = [[1, 1, 0.25],
                                    [1, 1, 0.75],
                                    [1.5, 1.5, 2],
                                    [0.5, 1.5, 0.5],
                                    [0.5, 3, 1.5],
                                    [0.5, 2, 1.5]
                                    ]
    obstacle_dic['ruler'] = [[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]]



    path = [[[0.5, 0.2, 0.6], [0.6, 0.3, 0.6], [0.7, 0.4, 0.6], [0.8, 0.5, 0.6], [0.9, 0.6, 0.6], [1.0, 0.6, 0.6],
             [1.1, 0.6, 0.6], [1.2, 0.6, 0.6], [1.3, 0.6, 0.6], [1.4, 0.7, 0.6], [1.5, 0.8, 0.6], [1.5, 0.9, 0.6],
             [1.5, 1.0, 0.6], [1.5, 1.1, 0.6], [1.5, 1.2, 0.6], [1.5, 1.3, 0.6], [1.5, 1.4, 0.6], [1.5, 1.5, 0.6],
             [1.5, 1.6, 0.6], [1.5, 1.7, 0.6], [1.5, 1.8, 0.6], [1.5, 1.9, 0.6], [1.5, 2.0, 0.6], [1.5, 2.1, 0.6],
             [1.5, 2.2, 0.5], [1.5, 2.2, 0.6]],
            [[0.2, 0.5, 0.6], [0.3, 0.6, 0.6], [0.4, 0.7, 0.6], [0.5, 0.8, 0.6], [0.6, 0.9, 0.6], [0.6, 1.0, 0.6],
             [0.6, 1.1, 0.7], [0.6, 1.1, 0.8], [0.6, 1.2, 0.9], [0.6, 1.3, 0.9], [0.7, 1.4, 0.9], [0.8, 1.5, 0.9],
             [0.9, 1.5, 0.8], [1.0, 1.5, 0.7], [1.1, 1.5, 0.6], [1.2, 1.5, 0.6], [1.3, 1.5, 0.6], [1.4, 1.5, 0.6],
             [1.5, 1.5, 0.6], [1.6, 1.5, 0.6], [1.7, 1.5, 0.6], [1.8, 1.5, 0.6], [1.9, 1.5, 0.6], [2.0, 1.5, 0.6],
             [2.1, 1.5, 0.6], [2.2, 1.5, 0.6]]]


    # initialize drawer and figure
    drawer = Drawer()
    fig = pv.figure()

    # draw obstacles and path
    fig = drawer.draw_obstacle(fig, obstacle_dic)
    fig = drawer.draw_path(fig, path, line_width=3)

    fig.show()
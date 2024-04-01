# Two-stage Multi-UAV Planning Solution Combining A*, Model Predictive Control, and Artificial Potential Field

This is the repository containing the codes used for final project of RO47005 Planning and Decision Making, group 18.

## Installation

### On *macOS* and *Ubuntu*

```sh
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

### On *Windows*

#### Requirements

Download Visual Studio and [C++ 14.0](https://visualstudio.microsoft.com/downloads/)
- We recommend the free Community version
- Select "Desktop development with C++"

Download [Python 3](https://www.python.org/downloads/release/python-390/)
- Note: we used the [Windows x86-64 installer](https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe) on Windows 10 Home

Download a Python IDE
- We recommend [PyCharm Community](https://www.jetbrains.com/pycharm/download/#section=windows)
- Select all the options in the installer and reboot

#### Installation

Download the code, unzip and open the project in PyCharm 

To run code you may need to configure PyCharm. 
- Go to `File->Settings` and Select `RO47005-pyll-final-project->Python Interpreter`

- Select the `+` 

> Type `numpy` and click "Install package".

> Type `matplotlib` and click "Install package".

> Type `pybullet` and click "Install package".

> Type `gym` and click "Install package".

> Type `Pillow` and click "Install package".

> Type `Cycler` and click "Install package".

> Type `cvxpy` and click "Install package".

> Type `pytransform3d` and click "Install package".

> Type `open3d` and click "Install package".

## Run

All demos are located in folder `gym_pybullet_drones/demo`

### AStarOne.py

An example of path planning for single UAV using modified A* algorithm.

### RRTOne.py

An example of path planning for single UAV using modified RRT* algorithm.

### moving_obstacle.py

An example of path planning for multiple UAVs using modified A* algorithm as global planner and MPC as local planner in simple scenario.

### SimpleAStarMul.py

An overall of path planning for multiple UAVs using A* algorithm as global planner and MPC combined with artificial potential field as local planner in complex scenario.

### ModifiedAStarMul.py

An overall of path planning for multiple UAVs using modified A* algorithm as global planner and MPC combined with artificial potential field as local planner in complex scenario.

## Demo

This [video](https://youtu.be/RkOqEFh1KFM) shows the simulation and result of the previous demos.

## Check for errors
If the results deviate from expectations, please turn off the artificial potential field in DSLPIDControl.py


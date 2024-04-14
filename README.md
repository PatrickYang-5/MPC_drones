# MPC for Multi-UAVs

This is the repository containing the Linear MPC codes used for Multi-UAVs.

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
- Go to `File->Settings` and Select `Your dictionary->Python Interpreter`

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

The demos for MPC are located in folder `gym_pybullet_drones/demo`, named `RegularMPC` and `OutputMPC` 

### RegularMPC.py

A sample program for testing regular MPC, which can be used for a single drone or multiple drones, but does not incorporate multi-drone interaction.

### OutputMPC.py

A sample program is used to test output MPC, which can be used for a single UAV or for multiple UAVs, where the constraints between the UAVs can be adjusted to soft and hard constraints.


## Contact
This program is tested in Linux 20.04 and MacOS with M1 chip, if there are solving errors please try to adjust the weight matrix, prediction intervals and soft constraint weights, or contact the author for feedback on the problem.


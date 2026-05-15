# MPC for Multi-UAVs

This repository provides a PyBullet-based simulation framework for multi-UAV
experiments involving three-dimensional path planning, PID trajectory tracking,
local and full-state linear model predictive control (MPC), and inter-agent
collision avoidance. The principal executable experiments are located in
`gym_pybullet_drones/demo/`.

## Installation

All commands below should be executed from the repository root.

```sh
conda create -n drones python=3.10
conda activate drones

python -m pip install --upgrade pip
python -m pip install -e .

# Additional packages required by the MPC and 3D visualization demos
python -m pip install cvxpy control pytransform3d open3d
```

On Ubuntu, if `pybullet` cannot be built because of missing compilation tools,
install the required build utilities first:

```sh
sudo apt install build-essential
```

On Windows, Python 3.10 and the Visual Studio C++ Build Tools are recommended
before running the same `pip` commands.

## Repository Structure

```text
gym_pybullet_drones/
  demo/          Executable experiment scripts
  PathPlanning/  AStar, EasyAStar, RRT, and local MPC path utilities
  MPC/           Linear MPC dynamics, terminal-set computation, full-state MPC
  control/       PID and MPC-PID controllers
  envs/          PyBullet UAV simulation environments
  utils/         Logging, visualization, command-line helpers, enumerations
results/         Default output directory for logs, CSV files, figures, videos
```

## Running Experiments

Most experiments can be executed directly. For example:

```sh
python gym_pybullet_drones/demo/AStarOne.py
```

Scripts that expose command-line arguments generally support the following
options:

```sh
--gui true|false
--plot true|false
--record_video true|false
--duration_sec 10
--output_folder results
```

Example:

```sh
python gym_pybullet_drones/demo/AStarOne.py --duration_sec 10 --record_video false
```

Boolean arguments accept `true/false`, `yes/no`, `1/0`, or `t/f`.

## Experiment Descriptions

### Global Path Planning with PID Tracking

| Experiment | Description | Command |
| --- | --- | --- |
| `AStarOne.py` | Single-UAV three-dimensional global path planning followed by PID trajectory tracking. Although the script name refers to A*, the current default planner is `EasyAStar`. The script reports planning time and path length, opens a 3D path visualization, and saves flight logs. | `python gym_pybullet_drones/demo/AStarOne.py` |
| `RRTOne.py` | Single-UAV RRT-based global path planning followed by PID trajectory tracking. This experiment is useful for comparing RRT path geometry, path length, and control effort with A*-based planners. | `python gym_pybullet_drones/demo/RRTOne.py` |
| `SimpleAStarMul.py` | Two-UAV EasyAStar planning with PID tracking. The default scenario contains two intersecting routes and is intended to evaluate multi-agent path generation and tracking behavior. | `python gym_pybullet_drones/demo/SimpleAStarMul.py` |
| `ModifiedAStarMul.py` | Two-UAV standard `AStar` planning with PID tracking. This experiment provides a comparison point for the simplified EasyAStar implementation used in `SimpleAStarMul.py`. | `python gym_pybullet_drones/demo/ModifiedAStarMul.py` |
| `moving_obstacle.py` | Three-UAV scenario in which two UAVs use local position-level MPC for path adjustment while the third UAV acts as a moving obstacle or agent. This experiment is intended for dynamic obstacle-avoidance evaluation. | `python gym_pybullet_drones/demo/moving_obstacle.py` |

These experiments rely primarily on `CtrlAviary`, `DSLPIDControl`, `GlobalMap`,
`GlobalPathPlanning`, and `Drawer`. By default, logs are written to `results/`.
Several scripts call `fig.show()` for the 3D path viewer and therefore require
a graphical desktop session.

### Model Predictive Control Experiments

| Experiment | Description | Command |
| --- | --- | --- |
| `RegularMPC.py` | Linear MPC state-propagation experiment for a single UAV by default. It uses `Whole_UAV_dynamics` and `LMPC`, draws the simulated trajectory in PyBullet, and plots position tracking against a fixed target. This script calls `run()` directly and does not parse command-line arguments. | `python gym_pybullet_drones/demo/RegularMPC.py` |
| `OutputMPC.py` | Main multi-UAV output MPC experiment. The script first generates global paths, converts them into reference trajectories, and then applies full-state LMPC with soft inter-UAV separation constraints. The default scenario contains two UAVs with intersecting routes. This script calls `run()` directly and does not parse command-line arguments. | `python gym_pybullet_drones/demo/OutputMPC.py` |
| `EarlytypeMPC.py` | Earlier integrated PyBullet and MPC-PID experiment. It advances the PyBullet environment, solves an LMPC problem for force inputs, and passes the resulting force command to `MPCPIDControl`. This script is primarily useful as a reference for the previous control pipeline and produces extensive debug output. | `python gym_pybullet_drones/demo/EarlytypeMPC.py` |

To pass arguments to `RegularMPC.py` or `OutputMPC.py`, import and call `run()`
from the command line:

```sh
python -c "from gym_pybullet_drones.demo.OutputMPC import run; run(gui=False, plot=False, record_video=False, draw_debug_lines=False)"
```

```sh
python -c "from gym_pybullet_drones.demo.RegularMPC import run; run(gui=True, verbose=True)"
```

## Selecting an Experiment

Use `AStarOne.py`, `RRTOne.py`, `SimpleAStarMul.py`, or
`ModifiedAStarMul.py` to evaluate global path planning followed by PID
tracking. Use `moving_obstacle.py` to evaluate dynamic obstacle avoidance. Use
`RegularMPC.py` for a compact single-UAV linear MPC tracking test. Use
`OutputMPC.py` for the principal multi-UAV MPC experiment with soft inter-agent
separation constraints.

## Modifying Experimental Scenarios

Initial positions, target positions, planner selection, MPC horizon, and map
dimensions are generally defined near the beginning of each demo script or in
the first part of its `run()` function:

```python
INIT_XYZS = ...
INIT_RPYS = ...
GOAL = ...
GLOBAL_PLANNER_METHOD = "EasyAStar"  # "EasyAStar", "AStar", or "RRT"
```

If `--num_drones` is changed, the arrays `INIT_XYZS`, `INIT_RPYS`, and `GOAL`
must be updated accordingly so that their lengths match the number of UAVs.

Important default parameters are summarized below.

| Parameter | Meaning |
| --- | --- |
| `DEFAULT_GUI` | Enables the PyBullet graphical interface. |
| `DEFAULT_PLOT` | Enables trajectory or logger plots where supported. Some scripts do not route all figures through this flag. |
| `DEFAULT_RECORD_VISION` | Enables PyBullet frame or video recording. `OutputMPC.py` defaults to `True`. |
| `DEFAULT_DURATION_SEC` | Duration of experiments that step the PyBullet environment. |
| `DEFAULT_CONTROL_FREQ_HZ` | Controller frequency, default `48 Hz`. |
| `DEFAULT_SIMULATION_FREQ_HZ` | PyBullet simulation frequency, default `240 Hz`. |
| `DEFAULT_MPC_HORIZON` | Prediction horizon used by `OutputMPC.py`, default `6`. |

## Outputs

By default, simulation outputs are written to `results/`:

```text
results/
  save-flight-*.npy          # NumPy log file generated by Logger.save()
  save-flight-pid-*/         # CSV directory generated by Logger.save_as_csv("pid")
  recording_*/               # Video frames or recording output when record_video is enabled
  output_figure.png          # Figure saved by Logger.plot()
```

For a short non-recording trial:

```sh
python gym_pybullet_drones/demo/AStarOne.py --record_video false --duration_sec 5
```

For headless execution, prefer scripts whose `gui` and `plot` options can be
disabled. The path-planning demos still open a 3D window through
`pytransform3d.visualizer`; on machines without display support, the relevant
`fig.show()` calls should be disabled or a virtual display should be used.

## Core Modules

| Module | Purpose |
| --- | --- |
| `PathPlanning/GlobalMap.py` | Constructs the shared voxel map used by planners and environments. |
| `PathPlanning/GlobalPathPlanning.py` | Dispatches planning requests to `EasyAStar`, `AStar`, or `RRT`. |
| `PathPlanning/MPC.py` | Implements local position-level MPC for waypoint adjustment around obstacles or neighboring UAVs. |
| `MPC/LMPCHover.py` | Implements full-state linear MPC, including UAV dynamics, terminal-set integration, and soft inter-agent separation constraints. |
| `MPC/TerminalSet.py` | Computes and checks terminal constraints for the MPC problem. |
| `control/DSLPIDControl.py` | Baseline PID controller used by the path-planning experiments. |
| `control/MPCPIDControl.py` | PID controller variant that can incorporate MPC-derived force inputs. |
| `utils/Logger.py` | Saves `.npy` logs, CSV files, and figures. |
| `utils/Drawer.py` | Visualizes obstacles and planned paths in the 3D viewer. |

## Troubleshooting

- `ModuleNotFoundError: No module named 'control'`: run `python -m pip install control`.
- `ModuleNotFoundError: No module named 'cvxpy'`: run `python -m pip install cvxpy`.
- `pytransform3d` or 3D viewer errors: run `python -m pip install pytransform3d open3d`.
- MPC solver failures or infeasible problems are often caused by an overly long
  prediction horizon, overly restrictive constraints, unsuitable weights, or
  inconsistent initial and target states.

## Notes

The repository has been tested on Ubuntu 20.04 and macOS, including Apple
Silicon. If an MPC instance becomes infeasible, first verify the weighting
matrices, prediction horizon, soft-constraint weights, and consistency between
the hard-coded number of UAVs, initial states, and target states.

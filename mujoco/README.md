# Crazyflie Simulation with MuJoCo

This repository contains code to simulate the Crazyflie quadcopter using MuJoCo. Follow the instructions below to set up and run the simulation.

## Installation

1. Clone the repository:
```sh
    git clone https://github.com/yourusername/crazyflie-sim.git
    git submodule sync
    git submodule update --init --recursive
```
2. Build the python bindings of the crazyflie-firmware
```sh
    cd path/to/crazyflie-firmware
    make cf2_defconfig
    make -j
    make bindings_python
    export PYTHONPATH=path/to/crazyflie-sim/deps/crazyflie-firmware/build:$PYTHONPATH
```
## Usage

To run the simulation, execute the following command:
```sh
    cd crazyflie-sim/mujoco/scripts/
    python3 model.py --traj_path ../../data/figure8_traj.csv --models_path ../../deps/dynobench/models/quad3d_v0.yaml  
```
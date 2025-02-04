import numpy as np
import argparse 
from pathlib import Path
from helper import *
import pydynobench as pydy
import cffirmware as cff
import math
import rowan as rn
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

def setSetpoint():
    setpoint = cff.setpoint_t()
    setpoint.mode.x = cff.modeAbs
    setpoint.mode.y = cff.modeAbs
    setpoint.mode.z = cff.modeAbs
    setpoint.mode.roll = cff.modeDisable
    setpoint.mode.pitch = cff.modeDisable
    setpoint.mode.yaw = cff.modeAbs
    return setpoint

def setupCtrl(model_params):
    ctrlLee = cff.controllerLee_t()
    cff.controllerLeeInit(ctrlLee)
    state = cff.state_t()
    sensors = cff.sensorData_t()
    control = cff.control_t()
    setpoint = setSetpoint()
    ctrlLee.mass = model_params["m"]
    return ctrlLee, state, control, setpoint, sensors


def updateState(st, state, sensors):
    state.position.x = st[0]  # m
    state.position.y = st[1]  # m
    state.position.z = st[2]  # m
  
    state.attitudeQuaternion.x = st[3]
    state.attitudeQuaternion.y = st[4]
    state.attitudeQuaternion.z = st[5]
    state.attitudeQuaternion.w = st[6]

    state.velocity.x = st[7]  # m/s
    state.velocity.y = st[8]  # m/s
    state.velocity.z = st[9]  # m/s

    sensors.gyro.x = np.degrees(st[10])  # deg/s
    sensors.gyro.y = np.degrees(st[11])  # deg/s
    sensors.gyro.z = np.degrees(st[12])  # deg/s

    return state, sensors

def updateSetpoint(traj, setpoint, states_d):
    pos  = traj[ 1 :  4]
    vel  = traj[ 4 :  7]
    acc  = traj[ 7 : 10]
    jerk = traj[10 : 13]
    snap = traj[13 : 16]

    setpoint.position.x = pos[0]  # m
    setpoint.position.y = pos[1]  # m
    setpoint.position.z = pos[2]  # m

    setpoint.velocity.x = vel[0]  # m/s
    setpoint.velocity.y = vel[1]  # m/s
    setpoint.velocity.z = vel[2]  # m/s

    setpoint.acceleration.x = acc[0]  # m/s^2
    setpoint.acceleration.y = acc[1]  # m/s^2
    setpoint.acceleration.z = acc[2]  # m/s^2

    setpoint.jerk.x = jerk[0]  # m/s^3
    setpoint.jerk.y = jerk[1]  # m/s^3
    setpoint.jerk.z = jerk[2]  # m/s^3

    setpoint.snap.x = snap[0]  # m/s^4
    setpoint.snap.y = snap[1]  # m/s^4
    setpoint.snap.z = snap[2]  # m/s^4

    yaw_d = traj[16] # rad

    setpoint.attitude.yaw = 0.0 # rad
    setpoint.attitudeRate.yaw = 0.0 # rad/s
    setpoint.attitudeAcc.yaw = 0.0 # rad/s^2

    states_d[0:3]   = traj[1 : 4]
    states_d[6:9]   = traj[4 : 7]
    return setpoint, states_d


def setInitState(states, traj):
    states[0] = [traj[0,1], traj[0,2], traj[0,3], 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    return states


def getDesiredStates(ctrlLee, states_d, traj):
    rpy_des         = np.zeros(3,)
    omega_d         = np.zeros(3,)
    omega_dd_des    = np.zeros(3,)
    rpy_des[0]      = ctrlLee.rpy_des.x
    rpy_des[1]      = ctrlLee.rpy_des.y
    rpy_des[2]      = ctrlLee.rpy_des.z
    omega_d[0]      = ctrlLee.omega_r.x
    omega_d[1]      = ctrlLee.omega_r.y
    omega_d[2]      = ctrlLee.omega_r.z
    omega_dd_des[0] = ctrlLee.omega_des_dot.x
    omega_dd_des[1] = ctrlLee.omega_des_dot.y
    omega_dd_des[2] = ctrlLee.omega_des_dot.z
    states_d[0:3]   = traj[ 1:4]
    states_d[3:6]   = rpy_des
    states_d[6:9]   = traj[4:7]
    states_d[9:12]  = omega_d
    states_d[12:15] = omega_dd_des
    return states_d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj',required=True, type=str, help="problem description")
    parser.add_argument('--models_path',required=True, type=str, help="problem description")
    args = parser.parse_args()
    
    # model parameters
    model_params = loadyaml(args.models_path) # dynobench/models/quad3d_v0.yaml
    # dt = model_params["dt"]
    dt = model_params["dt"]
    arm_length = model_params["arm_length"]
    arm = 0.707106781 * arm_length
    t2t = model_params["t2t"]
    # map u between [0,max_f], max_t = 1.4
    u_nominal = model_params["m"] * 9.81 / 4
    # allocation matrix
    B0 = u_nominal * np.array(
        [[1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t],])
    B0_inv = np.linalg.inv(B0)

    # robot object
    robot =  pydy.robot_factory(args.models_path , [-1000, -1000, -1.0], [1000, 1000, 1.0])
    
    # load reference trajectory: pos, vel, acc (w/o grav), jerk , snap
    traj = loadcsv(args.traj)
    T = traj[-1,0] # final time
    ts = traj[:,0]
    # controller setup
    ctrlLee, state, control, setpoint, sensors = setupCtrl(model_params)
    
    # logging states
    states = np.zeros((ts.shape[0],13)) # states according to dynobench [x y z qx qy qz qw vx vy vz wx wy wz]
    states_d = np.zeros((ts.shape[0], 15)) # desired values: pos, rpy, vel, omega, omega_dot 
    actions = np.zeros((ts.shape[0] - 1, 4))

    rpy = np.zeros((ts.shape[0],3))
    # set initial state    
    states = setInitState(states, traj)
    for k, t in enumerate(ts):
        state, sensors = updateState(states[k], state, sensors)
        setpoint, states_d[k] = updateSetpoint(traj[k], setpoint, states_d[k])
        if t < T:
            cff.controllerLee(ctrlLee, control, setpoint, sensors , state, 0)
            eta = np.array([control.thrustSi, control.torqueX, control.torqueY, control.torqueZ])
            rpy[k] = np.array([ctrlLee.rpy.x, ctrlLee.rpy.y, ctrlLee.rpy.z])
            states_d[k] = getDesiredStates(ctrlLee, states_d[k], traj[k])
            u = B0_inv@eta
            robot.step(states[k + 1], states[k], u, dt)

    # plotting (pos, vel, rpy, omega, omega_dot, acc, jerk, snap)
    pos = states[:,0:3]
    vel = states[:,7:10]
    omega = states[:,10:13]

    pos_d    = states_d[:, 0:3]
    vel_d    = states_d[:, 6:9]
    acc_d  = traj[:,  7 : 10]
    jerk_d = traj[:, 10 : 13]
    snap_d = traj[:, 13 : 16]
    rpy_des  = states_d[:, 3:6] 
    omega_d  = states_d[:, 9:12]
    omega_dd  = states_d[:, 12:15]


    fig, ax = plt.subplots(5, 3, sharex='all')
    for k, axis in enumerate(["x", "y", "z"]):    
        ax[0,k].plot(ts, pos[:,k], label="act")
        ax[0,k].plot(ts, pos_d[:,k], label="ref")
        ax[0,k].set_ylabel(f"pos {axis}[m]")

        ax[1,k].plot(ts, vel[:,k])
        ax[1,k].plot(ts, vel_d[:,k])
        ax[1,k].set_ylabel(f"vel {axis}[m/s]")

        ax[2,k].plot(ts, np.degrees(rpy[:,k]))
        ax[2,k].plot(ts, np.degrees(rpy_des[:,k]))
        ax[2,k].set_ylabel(f"rot {axis} [deg]")

        ax[3,k].plot(ts, np.degrees(omega[:,k]))
        ax[3,k].plot(ts, np.degrees(omega_d[:,k]))
        ax[3,k].set_ylabel(f"ang vel {axis}[deg/s]")

        ax[4,k].plot(ts, np.degrees(omega_dd[:,k]), color="darkorange")
        ax[4,k].set_ylabel(f"ang acc {axis}[deg/s^2]")


        # ax[5,k].plot(ts, acc_d[:,k], color="darkorange")
        # ax[5,k].set_ylabel(f"acc {axis}[m/s^2]")

        # ax[6,k].plot(ts, jerk_d[:, k], color="darkorange")
        # ax[6,k].set_ylabel(f"jerk {axis}[m/s^3]")

        # ax[7,k].plot(ts, snap_d[:, k], color="darkorange")
        # ax[7,k].set_ylabel(f"snap {axis}[m/s^4]")

    ax[0,0].legend()
    plt.show()


if __name__ == "__main__":
    main()
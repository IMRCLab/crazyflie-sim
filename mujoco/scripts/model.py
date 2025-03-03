import mujoco as mj
import numpy as np
import time
from mujoco.glfw import glfw
import cffirmware as cff
from helper import *
import argparse 


np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)



class LeeController():
    def __init__(self, model_params):
        """Initialize the Lee controller."""
        self.ctrlLee = cff.controllerLee_t()
        cff.controllerLeeInit(self.ctrlLee)
        self.state = cff.state_t()
        self.sensors = cff.sensorData_t()
        self.control = cff.control_t()
        self.ctrlLee.mass = model_params["m"]
        self.setSetpoint()
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.B0_inv = np.linalg.inv(self.B0)
    def setSetpoint(self):
        """Set the desired setpoint modes."""
        self.setpoint = cff.setpoint_t()
        self.setpoint.mode.x = cff.modeAbs
        self.setpoint.mode.y = cff.modeAbs
        self.setpoint.mode.z = cff.modeAbs
        self.setpoint.mode.roll = cff.modeDisable
        self.setpoint.mode.pitch = cff.modeDisable
        self.setpoint.mode.yaw = cff.modeAbs

    def updateState(self, st):
        """Update the drone state in the firmware controller."""
        self.state.position.x = st[0]
        self.state.position.y = st[1]
        self.state.position.z = st[2]
        print(st)
        self.state.attitudeQuaternion.w = st[3]
        self.state.attitudeQuaternion.x = st[4]
        self.state.attitudeQuaternion.y = st[5]
        self.state.attitudeQuaternion.z = st[6]

        self.state.velocity.x = st[7]
        self.state.velocity.y = st[8]
        self.state.velocity.z = st[9]

        self.sensors.gyro.x = np.degrees(st[10])
        self.sensors.gyro.y = np.degrees(st[11])
        self.sensors.gyro.z = np.degrees(st[12])


    def updateSetpoint(self, traj):
        """Update the desired trajectory setpoint."""
        pos = traj[1:4]
        vel = traj[4:7]
        acc = traj[7:10]
        jerk = traj[10:13]
        snap = traj[13:16]

        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]

        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]

        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.jerk.x = jerk[0]
        self.setpoint.jerk.y = jerk[1]
        self.setpoint.jerk.z = jerk[2]

        self.setpoint.snap.x = snap[0]
        self.setpoint.snap.y = snap[1]
        self.setpoint.snap.z = snap[2]

        self.setpoint.attitude.yaw = 0.0 # rad
        self.setpoint.attitudeRate.yaw = 0.0 # rad/s
        self.setpoint.attitudeAcc.yaw = 0.0 # rad/s^2


    def getControl(self):
        """Compute the control command using the Lee controller."""
        print("position setpoint: ",self.setpoint.position.x, self.setpoint.position.y, self.setpoint.position.z)
        print("velocity setpoint: ",self.setpoint.velocity.x, self.setpoint.velocity.y, self.setpoint.velocity.z)
        print("position state: ",self.state.position.x, self.state.position.y, self.state.position.z)
        print("velocity state: ",self.state.velocity.x, self.state.velocity.y, self.state.velocity.z)


        cff.controllerLee(self.ctrlLee, self.control, self.setpoint, self.sensors, self.state, 0)


        eta = np.array([self.control.thrustSi, self.control.torqueX, self.control.torqueY, self.control.torqueZ])
        return self.B0_inv @ eta

class CFMujoco():
    def __init__(self, xml_path, name, sim_args):
        """Initialize the MuJoCo simulation."""
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed!")

        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.model.opt.timestep = 0.01
        # Load trajectory and model parameters
        self.model_params = sim_args["model_params"]
        self.traj_path = sim_args["traj_path"]
        self.traj = loadcsv(self.traj_path)
        self.T = self.traj[-1, 0]
        self.ts = self.traj[:, 0]

        # Create Window
        self.window = glfw.create_window(1200, 900, name, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed!")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.show_window(self.window)

        # Initialize visualization
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Install input callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)

        # Mouse tracking
        self.button_left = False
        self.button_right = False
        self.last_x, self.last_y = 0, 0

        self.controller = LeeController(self.model_params)
        self.setInitState()
        self.controller.updateSetpoint(self.traj[0])

    def setInitState(self):
        """Set the initial state of the drone."""
        self.data.qpos[:] = [self.traj[0,1], self.traj[0,2], self.traj[0,3], 1, 0, 0, 0]
        self.data.qvel[:] = [0, 0, 0, 0, 0, 0]
        self.controller.updateState(np.concatenate((self.data.qpos, self.data.qvel)))

    def keyboard(self, window, key, scancode, action, mods):
        """Handle keyboard events."""
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def mouse_button(self, window, button, action, mods):
        """Handle mouse button events."""
        if action == glfw.PRESS:
            self.button_left = (button == glfw.MOUSE_BUTTON_LEFT)
            self.button_right = (button == glfw.MOUSE_BUTTON_RIGHT)
            self.last_x, self.last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self.button_left = self.button_right = False

    def mouse_move(self, window, xpos, ypos):
        """Handle mouse movement for camera control."""
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if self.button_left:
            mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ROTATE_H, dx / 100, dy / 100, self.scene, self.cam)
        elif self.button_right:
            mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, dx / 100, dy / 100, self.scene, self.cam)

    def scroll(self, window, xoffset, yoffset):
        """Handle scroll events for zooming."""
        mj.mjv_moveCamera(self.model, mj.mjtMouse.mjMOUSE_ZOOM, 0, yoffset / 10, self.scene, self.cam)

    def simulate(self):
        """Run the simulation loop with real-time updates."""
        
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            for k, t in enumerate(self.ts):
                dt = self.model.opt.timestep
                self.controller.updateSetpoint(self.traj[k])
                eta = self.controller.getControl()
                self.data.ctrl[:] = np.array(eta)

                mj.mj_step(self.model, self.data)
                self.controller.updateState(np.concatenate((self.data.qpos, self.data.qvel)))

                # get framebuffer viewport
                viewport_width, viewport_height = glfw.get_framebuffer_size(
                    self.window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                # Update scene and render
                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                    mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                mj.mjr_render(viewport, self.scene, self.context)

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(self.window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()
                # time.sleep(0.1)        
        glfw.terminate()
    
      
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_path', required=True, type=str, help="Path to trajectory CSV file")
    parser.add_argument('--models_path', required=True, type=str, help="Path to model parameters YAML file")
    args = parser.parse_args()
    
    # Load model parameters
    model_params = loadyaml(args.models_path)
    sim_args = {"model_params": model_params, "traj_path": args.traj_path}

    # Initialize and run simulation
    xml_path = "../models/xml/crazyflie.xml"
    sim = CFMujoco(xml_path, "Crazyflie Simulation", sim_args)
    sim.simulate()

if __name__ == "__main__":
    main()

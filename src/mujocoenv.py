import mujoco
import numpy as np
from mujoco import viewer
import time

class MujocoEnv:
    def __init__(self, model_path):
        # diff link angle from vertical axis
        self.alpha1 = np.arctan(82/316)
        self.alpha2 = -np.arctan(82/386)
        self.alpha3 = -np.arctan(88/107)

        # diff link angle from joint local axis
        self.gamma1 = self.alpha1
        self.gamma2 = self.alpha2 - self.alpha1
        self.gamma3 = self.alpha3 - self.alpha2

        self.current_transform = [0.2, 0.1, np.pi]  # x, z, pitch

        self.initialize(model_path)
        self.set_initial_transform()
        # time.sleep(2)

    def initialize(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 120, 120)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.v = viewer.launch_passive(self.model, self.data)

    def step(self):
        if self.v.is_running():
            mujoco.mj_step(self.model, self.data)
            self.v.sync()

        else:
            self.v.close()

    def foward_kinematics_3axis(self, q1, q2, q3):
        l1 = 0.3264659247149693
        l2 = 0.39265761166695853
        l3 = 0.13853880322855397
        q1 += self.gamma1
        q2 += self.gamma2
        q3 += self.gamma3
        x = l1*np.sin(q1)+l2*np.sin(q1+q2)+l3*np.sin(q1+q2+q3)
        z = l1*np.cos(q1)+l2*np.cos(q1+q2)+l3*np.cos(q1+q2+q3)
        pitch = q1 + q2 + q3

        return x, z, pitch

    # def step(self):
    #     mujoco.mj_step(self.model, self.data)

    def inverse_kinematics_3axis(self, x, z, pitch):
        l1 = 0.3264659247149693
        l2 = 0.39265761166695853
        l3 = 0.13853880322855397

        pitch = pitch + self.alpha3

        xw = x - l3* np.sin(pitch)
        zw = z - l3* np.cos(pitch)
        # print('xw: {}, zw: {}'.format(xw, zw))
        D = (xw**2 + zw**2 - l1**2 - l2**2) / (2 * l1 * l2)
        # print('D: {}'.format(D))

        q2 = np.arctan2(np.sqrt(1 - D**2), D) - self.gamma2
        q1 = np.arctan2(-l2 * np.sin(q2+self.gamma2), l1 + l2 * np.cos(q2+self.gamma2)) - np.atan2(zw, xw) + np.pi/2 - self.gamma1
        q3 = pitch - q1 - q2 - (self.gamma1+self.gamma2+self.gamma3)

        return q1, q2, q3
    
    def set_initial_transform(self, pos = [200, 0, 200], rot = [0, 180, 0], gripper=0):
        x = pos[0]*0.001
        z = pos[2]*0.001
        pitch = rot[1] * np.pi / 180

        self.set_joint_transform(x, z, pitch, gripper)

    def set_joint_transform(self, x, z, pitch, gripper):
        q1, q2, q3 = self.inverse_kinematics_3axis(x, z, pitch)
        # print(' q1: {}, q2: {}, q3: {}'.format(q1, q2, q3))
        x, y, z = self.foward_kinematics_3axis(q1, q2, q3)
        print(x, y, z)
        
        self.data.ctrl[1] = q1
        self.data.ctrl[3] = -q2
        self.data.ctrl[5] = np.pi - q3
        self.data.ctrl[6] = np.pi/4
        self.data.ctrl[7] = gripper  # gripper

    def set_transform_with_controller(self, button_state: dict):
        pitch = np.pi

        if button_state["+x"] == 1:
            self.current_transform[0] += 0.0001
        if button_state["-x"] == 1:
            self.current_transform[0] -= 0.0001
        if button_state["+z"] == 1:
            self.current_transform[1] += 0.0001
        if button_state["-Z"] == 1:
            self.current_transform[1] -= 0.0001
        if button_state["+pitch"] == 1:
            self.current_transform[2] += 0.0008
        if button_state["-pitch"] == 1:
            self.current_transform[2] -= 0.0008

        self.set_joint_transform(self.current_transform[0], self.current_transform[1], self.current_transform[2], 255- button_state["gripper"]*255)

    def get_camera_rgb(self):
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()

if __name__ == "__main__":
    # model_path = "models/franka_emika_panda/scene.xml"
    model_path = "models/universal_robots_ur5e/scene.xml"
    model = MujocoEnv(model_path)

    time.sleep(1)
    try:
        while True:
            model.step()
        
    except KeyboardInterrupt:
        model.v.close()
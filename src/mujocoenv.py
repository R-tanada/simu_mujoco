import mujoco
import numpy as np
from mujoco import viewer
import time
from actorcritic import ImagePreprocessor

class MujocoEnv:
    def __init__(self, model_path):
        self.process = ImagePreprocessor()

        # diff link angle from vertical axis
        self.alpha1 = np.arctan(82/316)
        self.alpha2 = -np.arctan(82/386)
        self.alpha3 = -np.arctan(88/107)

        # diff link angle from joint local axis
        self.gamma1 = self.alpha1
        self.gamma2 = self.alpha2 - self.alpha1
        self.gamma3 = self.alpha3 - self.alpha2

        self.current_transform = [0.4, 0.1, np.pi]  # x, z, pitch

        self.initialize(model_path)
        self.set_initial_transform()
        # time.sleep(2)

    def initialize(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 256, 256)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.v = viewer.launch_passive(self.model, self.data)

    def step(self, action):
        q1, q2, q3, q4 = action[0][0].item(), action[0][1].item(), action[0][2].item(), action[0][3].item()
        self.set_joint_transform(q1, q2, q3, q4)

        if self.v.is_running():
            mujoco.mj_step(self.model, self.data)
            self.v.sync()

        else:
            self.v.close()

        next_state = self.process.preprocess_image(self.get_camera_rgb())
        body = self.data.body('hand')
        body_pos = body.xpos
        body_quat = body.xquat
        reward = self.reward(body_pos)
        done, reward_2 = self.cheak_area(body_pos, body_quat)

        return next_state, reward+reward_2, done

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
    
    def set_initial_transform(self, pos = [400, 0, 0], rot = [0, 180, 0], gripper=0):
        x = pos[0]*0.001
        z = pos[2]*0.001
        pitch = rot[1] * np.pi / 180

        self.set_joint_transform_inv(x, z, pitch, gripper)

    def set_joint_transform_inv(self, x, z, pitch, gripper):
        q1, q2, q3 = self.inverse_kinematics_3axis(x, z, pitch)
        # print(' q1: {}, q2: {}, q3: {}'.format(q1, q2, q3))
        
        self.data.ctrl[1] = q1
        self.data.ctrl[3] = -q2
        self.data.ctrl[5] = np.pi - q3
        self.data.ctrl[6] = np.pi/4
        self.data.ctrl[7] = gripper  # gripper

    def set_joint_transform(self, r1, r2, r3, r4):
        q1, q2, q3 = self.inverse_kinematics_3axis(0.4, 0, np.pi)
        # print(' q1: {}, q2: {}, q3: {}'.format(q1, q2, q3))
        
        self.data.ctrl[1] = q1 + r1
        self.data.ctrl[3] = -q2 + r2
        self.data.ctrl[5] = np.pi - q3 + r4
        self.data.ctrl[7] = 255 + r4  # gripper

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
    
    def reset(self):
        self.set_initial_transform()
        for i in range(200):
            mujoco.mj_step(self.model, self.data)

        self.renderer.update_scene(self.data, camera=0)
        return self.process.preprocess_image(self.renderer.render())
    
    def reward(self, current_pos):
        box_pos = [0.5, 0, 0.02]
        return -np.linalg.norm(current_pos-box_pos)*10
    
    def cheak_area(sefl, pos, quat):
        def quat_to_euler_wxyz(quat):
            """
            クオータニオン [w, x, y, z] からオイラー角に変換
            返り値: roll (X軸回転), pitch (Y軸回転), yaw (Z軸回転)
            """
            w, x, y, z = quat

            # roll (X軸回転)
            t0 = +2.0 * (w*x + y*z)
            t1 = +1.0 - 2.0 * (x*x + y*y)
            roll = np.arctan2(t0, t1)

            # pitch (Y軸回転)
            t2 = +2.0 * (w*y - z*x)
            t2 = np.clip(t2, -1.0, 1.0)  # 数値安定性のためクリップ
            pitch = np.arcsin(t2)

            # yaw (Z軸回転)
            t3 = +2.0 * (w*z + x*y)
            t4 = +1.0 - 2.0 * (y*y + z*z)
            yaw = np.arctan2(t3, t4)

            return roll, pitch, yaw
        
        euler = quat_to_euler_wxyz(quat)
        box_pos = [0.5, 0, 0.02]

        if pos[0] < 0.35 or pos[0] > 0.6:
            return True, -100
        elif pos[2] < 0.12 or pos[2] > 0.5:
            return True, -100
        elif euler[1] < -0.45 or euler[1] > 0.45:
            return True, -100
        elif np.linalg.norm(pos - box_pos)<0.13:
            return True, 100
        
        return False, 0

    
# class RLEnv:
#     def __init__(self):
#         self.box_pos = [0.4, 0, 0.02]

#     def reward(self, current_pos):
        


if __name__ == "__main__":
    model_path = "models/franka_emika_panda/scene.xml"
    # model_path = "models/friction.xml"
    model = MujocoEnv(model_path)

    time.sleep(1)
    try:
        while True:
            model.step()
        
    except KeyboardInterrupt:
        model.v.close()
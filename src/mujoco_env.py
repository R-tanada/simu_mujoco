import mujoco
import numpy as np
from mujoco import viewer
import time
from actorcritic import ImagePreprocessor
import threading

class MujocoEnv:
    def __init__(self, model_path, visualize:bool = True):

        q1_range = [-1.7628, 1.7628]
        q2_range = [-3.0718, -0.0698]
        q3_range = [-0.0175, 3.7525]
        q4_range = [0, 255]
        self.q1_m = (q1_range[0]+q1_range[1])/2
        self.q2_m = (q2_range[0]+q2_range[1])/2
        self.q3_m = (q3_range[0]+q3_range[1])/2
        self.q4_m = (q4_range[0]+q4_range[1])/2

        self.current_transform = [0.4, 0.1, np.pi]  # x, z, pitch

        self.initialize(model_path, visualize)
        print('initialized Mujoco Environment')

    def initialize(self, model_path, visualize):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, 256, 256)

        if visualize == True:
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.v = viewer.launch_passive(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def step_custom_num(self, step_num):
        for i in range(step_num):
            mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_joint_transform(self, q1, q2, q3, q4):
        # q1: joint_1
        # q2: joint_2
        # q3: joint_3
        # q4: gripper
        self.data.ctrl[1] = q1 
        self.data.ctrl[3] = -q2
        self.data.ctrl[5] = np.pi - q3
        self.data.ctrl[6] = np.pi/4 # offset of the wrist joint
        self.data.ctrl[7] = q4

    def get_camera_rgb(self):
        self.renderer.update_scene(self.data, camera=0)
        return self.renderer.render()
    




    # def step(self, action):

    #     q1, q2, q3, q4 = action[0][0].item()*self.q1_m+self.q1_m, action[0][1].item()*self.q2_m+self.q2_m, action[0][2].item()*self.q3_m+self.q3_m, action[0][3].item()*self.q4_m+self.q4_m
    #     self.set_joint_transform(q1, q2, q3, q4)

    #     if self.v.is_running():
    #         mujoco.mj_step(self.model, self.data)
    #         self.v.sync()

    #     else:
    #         self.v.close()

    #     next_state = self.process.preprocess_image(self.get_camera_rgb())
    #     body = self.data.body('hand')
    #     body_pos = body.xpos
    #     body_quat = body.xquat
    #     reward = self.reward(body_pos)
    #     done, reward_2 = self.cheak_area(body_pos, body_quat)

    #     return next_state, reward+reward_2, done
    

    # def step(self, action):

    #     q1, q2, q3, q4 = action[0][0].item(), action[0][1].item(), action[0][2].item(), action[0][3].item()
    #     self.set_joint_transform(q1, q2, q3, q4)

    #     mujoco.mj_step(self.model, self.data)

    #     next_state = self.process.preprocess_image(self.get_camera_rgb())
    #     body = self.data.body('hand')
    #     body_pos = body.xpos
    #     body_quat = body.xquat
    #     reward = 1.5 + self.reward(body_pos)*10
    #     done, reward_2 = self.cheak_area(body_pos, body_quat)
    #     print('reward:', reward)

    #     return next_state, reward, done
    
    # def step(self, action):

    #     q1, q2, q3, q4 = action[0][0].item(), action[0][1].item(), action[0][2].item(), action[0][3].item()
    #     self.set_joint_transform(q1, q2, q3, q4)

    #     if self.v.is_running():
    #         mujoco.mj_step(self.model, self.data)
    #         self.v.sync()

    #     else:
    #         self.v.close()

    #     next_state = self.process.preprocess_image(self.get_camera_rgb())
    #     body = self.data.body('hand')
    #     body_pos = body.xpos
    #     body_quat = body.xquat
    #     reward = 2.5 + self.reward(body_pos)*10
    #     done, reward_2 = self.cheak_area(body_pos, body_quat)
    #     # print('reward:', reward)

    #     return next_state, reward, done

    
        # print(x, z, pitch, 255 + r4)

    # def set_joint_transform(self, r1, r2, r3, r4):
    #     q1, q2, q3 = self.inverse_kinematics_3axis(0.4, 0, np.pi)
    #     # print(' q1: {}, q2: {}, q3: {}'.format(q1, q2, q3))
        
    #     self.data.ctrl[1] = r1
    #     self.data.ctrl[3] = r2
    #     self.data.ctrl[5] = r3
    #     self.data.ctrl[7] = r4  # gripper






    
    def reward(self, current_pos):
        box_pos = [0.5, 0, 0.02]
        return -np.linalg.norm(current_pos-box_pos)
    
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

        if np.linalg.norm(pos - box_pos)<0.13:
            return True, 100
        
        return False, 0

    
# class RLEnv:
#     def __init__(self):
#         self.box_pos = [0.4, 0, 0.02]

#     def reward(self, current_pos):
        


if __name__ == "__main__":
    model_path = "models/franka_emika_panda/scene.xml"
    model = MujocoEnv(model_path)
    robot = RoboControl(model)

    buttom = Controller()
    buttom_thread = threading.Thread(target=buttom.get_joystick)
    buttom_thread.setDaemon(True)
    print('check')
    buttom_thread.start()
    

    try:
        while True:
            print(1)
            buttom_state = buttom.get_button_state()
            robot.set_transform_with_controller(buttom_state)
            model.step()
            model.v.sync()
        
    except KeyboardInterrupt:
        model.v.close()
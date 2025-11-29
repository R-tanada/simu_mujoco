from mujoco_env import MujocoEnv
import numpy as np
from mujoco import viewer
import time
# from actor_critic import ImagePreprocessor
from controller import Controller
import threading

class RoboControl:
    def __init__(self, mujo_env, controller:bool = False):
        self.mujo = mujo_env

        # length of each link
        self.l0 = 0.333
        self.l1 = 0.3264659247149693
        self.l2 = 0.39265761166695853
        self.l3 = 0.13853880322855397

        # difference of link angle from vertical axis
        self.alpha1 = np.arctan(82/316)
        self.alpha2 = -np.arctan(82/386)
        self.alpha3 = -np.arctan(88/107)

        # difference link angle from joint local axis
        self.gamma1 = self.alpha1
        self.gamma2 = self.alpha2 - self.alpha1
        self.gamma3 = self.alpha3 - self.alpha2

        self.current_x = 400*0.001
        self.current_z = 300*0.001
        self.current_pitch = 180*np.pi/180
        self.current_q1, self.current_q2, self.current_q3 = self.inverse_kinematics_3dof(x=0.4, z=0.3, pitch=np.pi)
        self.current_q4 = self.current_gripper = 255

        self.set_initial_transform_cartesian(x=400, z=300, pitch=180, gripper=255)
        time.sleep(1)
        print('initialzed Robot')

    def set_initial_transform_joint_angle(self, q1, q2, q3, q4):
        # transform mj model
        self.mujo.set_joint_transform(q1, q2, q3, q4)
        self.mujo.step_custom_num(500)

    def set_initial_transform_cartesian(self, x, z, pitch, gripper):
        x = x*0.001
        z = z*0.001
        pitch = pitch * np.pi / 180
        q1, q2, q3 = self.inverse_kinematics_3dof(x, z, pitch)
        q4 = gripper

        # transform mj model
        self.mujo.set_joint_transform(q1, q2, q3, q4)
        self.mujo.step_custom_num(500)

    def set_transform_joint_angle(self, q1, q2, q3, q4):
        # transform mj model
        self.mujo.set_joint_transform(q1, q2, q3, q4)
        self.mujo.step()

    def set_transform_cartesian(self, x, z, pitch, gripper):
        x = x*0.001
        z = z*0.001
        pitch = pitch * np.pi / 180
        q1, q2, q3 = self.inverse_kinematics_3dof(x, z, pitch)
        q4 = gripper

        # transform mj model
        self.mujo.set_joint_transform(q1, q2, q3, q4)
        self.mujo.step()

    def set_transform_with_controller(self, button_state: dict):

        if button_state["+x"] == 1:
            self.current_x += 0.0002
        if button_state["-x"] == 1:
            self.current_x -= 0.0002
        if button_state["+z"] == 1:
            self.current_z += 0.0002
        if button_state["-Z"] == 1:
            self.current_z -= 0.0002
        if button_state["+pitch"] == 1:
            self.current_pitch += 0.0008
        if button_state["-pitch"] == 1:
            self.current_pitch -= 0.0008

        q1, q2, q3 = self.inverse_kinematics_3dof(self.current_x, self.current_z, self.current_pitch)
        q4 = 255 - buttom_state["gripper"] * 255

        self.mujo.set_joint_transform(q1, q2, q3, q4)
        self.mujo.step()

    def get_transform_cartesian(self):
        return self.foward_kinematics_3dof(self.current_q1, self.current_q2, self.current_q3)

    def inverse_kinematics_3dof(self, x, z, pitch):
        pitch = pitch + self.alpha3
        z = z - self.l0

        # calculate the position of joint_3
        xw = x - self.l3* np.sin(pitch)
        zw = z - self.l3* np.cos(pitch)
        D = (xw**2 + zw**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)

        # calculate each joint angle
        q2 = np.arctan2(np.sqrt(1 - D**2), D) - self.gamma2
        q1 = np.arctan2(-self.l2 * np.sin(q2+self.gamma2), self.l1 + self.l2 * np.cos(q2+self.gamma2)) - np.atan2(zw, xw) + np.pi/2 - self.gamma1
        q3 = pitch - q1 - q2 - (self.gamma1+self.gamma2+self.gamma3)

        return q1, q2, q3
    
    def foward_kinematics_3dof(self, q1, q2, q3):
        q1 += self.gamma1
        q2 += self.gamma2
        q3 += self.gamma3
        x = self.l1*np.sin(q1)+self.l2*np.sin(q1+q2)+self.l3*np.sin(q1+q2+q3)
        z = self.l1*np.cos(q1)+self.l2*np.cos(q1+q2)+self.l3*np.cos(q1+q2+q3) + self.l0
        pitch = q1 + q2 + q3

        return x, z, pitch
    
    def euler_to_quat_wxyz(self, roll, pitch, yaw):
        """
        オイラー角 (ラジアン) からクオータニオン [w, x, y, z] に変換
        roll  : X軸回転
        pitch : Y軸回転
        yaw   : Z軸回転
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def quat_to_euler_wxyz(self, quat):
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
            buttom_state = buttom.get_button_state()
            robot.set_transform_with_controller(buttom_state)
            model.step()
            model.v.sync()

            time.sleep(1)
            model.reset()
        
    except KeyboardInterrupt:
        model.v.close()
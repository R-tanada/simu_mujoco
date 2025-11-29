from mujoco_env import MujocoEnv
from robot_control import RoboControl
from actor_critic import ImagePreprocessor, Agent
import time
import numpy as np  
import cv2
import mujoco
import torch


model_path = "models/franka_emika_panda/scene.xml"
mujo_env = MujocoEnv(model_path, visualize=True)
robot_control = RoboControl(mujo_env)

initial_time = time.perf_counter()
img_process = ImagePreprocessor()
agent = Agent()
# buffer = ReplayBuffer(50, 20)

# buttom_thred = threading.Thread(target=controller.get_joystick)
# buttom_thred.setDaemon(True)
# buttom_thred.start()

loop_freq = 500  # Hz
loop_time = 1.0 / loop_freq
box_pos = [0.5, 0, 0.15]
total_reward_list = []
q1_range = [-1.7628, 1.7628]
q2_range = [-3.0718, -0.0698]
q3_range = [-0.0175, 3.7525]
q4_range = [0, 255]
q1_m = (q1_range[0]+q1_range[1])/2
q3_m = (q3_range[0]+q3_range[1])/2
q4_m = (q4_range[0]+q4_range[1])/2

try:
    for episode in range(100):
        robot_control.initailize()
        state = robot_control.get_transform_joint_angle()
        state = torch.tensor(state, dtype=torch.float32)
        state[0] /= 1.7628
        state[1] /= 3
        state[2] /= 3.7525
        state[3] /= 255.0
        done = False
        total_reward = 0
        start_time = time.perf_counter()

        while not done:
            elapsed_time = time.perf_counter() - start_time
            # print(f"elapsed_time: {elapsed_time:.3f} sec")
            # get current state
            action, mean = agent.get_action(state)

            # calculete  current joint angle
            robot_control.current_q1 += action[0].item()
            robot_control.current_q2 += action[1].item()
            robot_control.current_q3 += action[2].item()
            robot_control.current_q4 += action[3].item()
            robot_control.set_transform_joint_angle(robot_control.current_q1, robot_control.current_q2, robot_control.current_q3, robot_control.current_q4)

            # get next state
            next_state = robot_control.get_transform_joint_angle()
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state[0] /= 1.7628
            next_state[1] /= 3
            next_state[2] /= 3.7525
            next_state[3] /= 255.0
            x, z, pitch = robot_control.get_transform_cartesian()
            reward, done = agent.reward(np.array([x, 0, z]), np.array(box_pos), elapsed_time)

            agent.update(state, action, mean, reward, next_state, done)
            state = next_state
            total_reward += reward
            # time.sleep(0.1)

        total_reward_list.append(total_reward)
        if episode % 10 == 0 and episode != 0:
            print("episode:{}, ave_total_reward:{}".format(episode, sum(total_reward_list[-10:])/10))

except KeyboardInterrupt:
    mujo_env.v.close()
    print("finish mainloop")



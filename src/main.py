from mujoco_env import MujocoEnv
from robot_control import RoboControl
from actor_critic import ImagePreprocessor, Agent
import time
import numpy as np  
import cv2
import mujoco


model_path = "models/franka_emika_panda/scene.xml"
mujo_env = MujocoEnv(model_path)
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
box_pos = [0.5, 0, 0.02]

try:
    for episode in range(100):
        robot_control.set_initial_transform_cartesian(x=400, z=300, pitch=180, gripper=255)
        state = img_process.preprocess_image(mujo_env.get_camera_rgb())
        done = False
        total_reward = 0
        start_time = time.perf_counter()

        while not done:
            elapsed_time = time.perf_counter() - start_time
            # get current state
            action, mean = agent.get_action(state)

            # calculete  current joint angle
            robot_control.current_q1 += action[0][0].item()
            robot_control.current_q2 += action[0][1].item()
            robot_control.current_q3 += action[0][2].item()
            robot_control.current_q4 += action[0][3].item()
            robot_control.set_transform_joint_angle(robot_control.current_q1, robot_control.current_q2, robot_control.current_q3, robot_control.current_q4)

            # get next state
            next_state = state = img_process.preprocess_image(mujo_env.get_camera_rgb())
            current_pos = robot_control.get_transform_cartesian()
            reward, done = agent.reward(np.array(current_pos), np.array(box_pos), elapsed_time)

            agent.update(state, action, mean, reward, next_state, done)
            state = next_state
            total_reward += reward
            # time.sleep(0.1)

        print(total_reward)

except KeyboardInterrupt:
    mujo_env.v.close()
    print("finish mainloop")



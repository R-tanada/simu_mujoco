from mujocoenv import MujocoEnv
from actorcritic import PolicyNet, ImagePreprocessor, ValueNet, Agent, ReplayBuffer
# from controller import Controller
import threading
import time
import math
import numpy as np  
import cv2
import mujoco


model_path = "models/franka_emika_panda/scene.xml"
mujo_env = MujocoEnv(model_path)

initial_time = time.perf_counter()
process = ImagePreprocessor()
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
        state = mujo_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, mean = agent.get_action(state)

            next_state, reward, done = mujo_env.step(action)

            agent.update(state, action, mean, reward, next_state, done)
            state = next_state
            total_reward += reward
            # time.sleep(0.1)

        print(total_reward)

except KeyboardInterrupt:
    mujo_env.v.close()
    print("finish mainloop")



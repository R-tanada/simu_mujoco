import mujoco
import numpy as np
from mujoco import viewer
import time

t0 = time.time()
model = mujoco.MjModel.from_xml_path("test_mujoco/mujoco_menagerie/franka_fr3/fr3.xml")
data = mujoco.MjData(model)

data.ctrl[0] = 0.1  # joint1 にコマンド入力
data.ctrl[1] = -0.1 # joint2 にコマンド入力


with viewer.launch_passive(model, data) as v:
    while v.is_running():
        t = time.time() - t0
        data.ctrl[0] = 0.1*np.sin(t)     # 関節1に周期的な入力：ゲイン0.1
        data.ctrl[1] = 0.1*np.cos(t)  
        mujoco.mj_step(model, data)
        v.sync()
        # angle1 = data.sensordata[0]
        # velocity1 = data.sensordata[1]
        # angle2 = data.sensordata[2]
        # velocity2 = data.sensordata[3]
        # print(velocity1)
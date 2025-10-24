import mujoco
import numpy as np
from mujoco import viewer
import time

t0 = time.time()
model = mujoco.MjModel.from_xml_path("models/franka_fr3/scene.xml")
data = mujoco.MjData(model)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

with viewer.launch_passive(model, data) as v:
    time.sleep(1)
    while v.is_running():
        data.ctrl[0] = 30*np.pi/180
        mujoco.mj_step(model, data)
        v.sync()
        test_data = data.sensor('pos1').data
        print(test_data)
        print(type(test_data))
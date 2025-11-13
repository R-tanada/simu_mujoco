import mujoco
import numpy as np
from mujoco import viewer
import time

class Model:
    def __init__(self, model_path):
        self.initialize(model_path)

    def initialize(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

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

    def move_joint(self, joint_name, position):



if __name__ == "__main__":
    model_path = "reference/mujoco_menagerie/franka_emika_panda/scene.xml"
    model = Model(model_path)

    time.sleep(1)
    try:
        while True:
            model.step()
        
    except KeyboardInterrupt:
        model.v.close()
from mujocoenv import MujocoEnv
from controller import Controller
import threading
import time
import math
import numpy as np  

model_path = "models/franka_emika_panda/scene.xml"
mujo_env = MujocoEnv(model_path)

initial_time = time.perf_counter()
# controller = Controller()

# buttom_thred = threading.Thread(target=controller.get_joystick)
# buttom_thred.setDaemon(True)
# buttom_thred.start()

try:
    while True:
        elasped_time = time.perf_counter() - initial_time
        # ------------------------------------------------------
        # 使用例
        # ------------------------------------------------------
        # # 目標角度を rad で指定
        # q_targets = {
        #     "actuator1": math.radians(0),
        #     "actuator2": math.radians(-30),
        #     "actuator3": math.radians(0),
        #     "actuator4": math.radians(-170),
        #     "actuator5": math.radians(0),
        #     "actuator6": math.radians(140),
        #     "actuator7": math.radians(45),
        # }

        # ctrl_dict = mujo_env.calc_ctrl_for_targets(q_targets)

        # # mj_step に反映
        # for act_name, ctrl_value in ctrl_dict.items():
        #     act_id = mujo_env.model.actuator(act_name).id
        #     mujo_env.data.ctrl[act_id] = ctrl_value
        x = elasped_time * 10
        print(x)

        mujo_env.data.ctrl[6] = np.pi/4
        if elasped_time > 3:
            mujo_env.transform(0.3, x*0.001, np.pi/2)

        mujo_env.step()

except KeyboardInterrupt:
    mujo_env.v.close()
    print("finish mainloop")
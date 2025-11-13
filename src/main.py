from model import Model
from controller import Controller
import threading
import time

model_path = "reference/mujoco_menagerie/franka_emika_panda/scene.xml"
model = Model(model_path)
controller = Controller()

buttom_thred = threading.Thread(target=controller.get_joystick)
buttom_thred.setDaemon(True)
buttom_thred.start()

try:
    while True:
        print(controller.pushed_button)
        model.step()

except KeyboardInterrupt:
    model.v.close()
    print("finish mainloop")
import pygame
import time
from collections import defaultdict

class Controller:
    def __init__(self):
        self.button_num = {
            "+x": 11, # `↑ボタン`
            "-x": 12, # `↓ボタン`
            "+z": 3, # `△ボタン`
            "-Z": 0, # `✕ボタン`
            "+pitch": 10, # R2ボタン
            "-pitch": 9, #L2ボタン
            "gripper": 5 # `R2ボタン`
        }
        self.button_state = defaultdict(lambda: 0)
        self.initialize()

    def initialize(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"接続中のコントローラー: {self.joystick.get_name()}")

        # 全ボタン数と軸数
        self.num_buttons = self.joystick.get_numbuttons()
        self.num_axes = self.joystick.get_numaxes()
        print(f"ボタン数: {self.num_buttons}")
        print(f"軸数: {self.num_axes}")

        # 各ボタンの状態を辞書で管理
        self.buttons = {
            i: {"prev": 0, "start_time": None, "long_pressed": False}
            for i in range(self.num_buttons)
        }

    # def get_joystick(self):
    #     try:
    #         while True:
    #             pygame.event.pump()

    #             # --- ボタン入力 ---
    #             for i in range(self.num_buttons):
    #                 self.pushed_button = self.joystick.get_button(i)
    #                 if self.pushed_button == 1:
    #                     self.buttons[i]["start_time"] = time.time()
    #                     self.buttons[i]["long_pressed"] = False
    #                     print(f"[ボタン{i}] 押下中")

    #             l2_value = self.joystick.get_axis(4)
    #             r2_value = self.joystick.get_axis(5)

    #             l2_normalized = (l2_value + 1) / 2  # 0.0〜1.0 に正規化
    #             r2_normalized = (r2_value + 1) / 2

    #             if l2_normalized > 0.05:
    #                 print(f"L2 押し込み量: {l2_normalized:.2f}")
    #             if r2_normalized > 0.05:
    #                 print(f"R2 押し込み量: {r2_normalized:.2f}")

    #             pygame.time.wait(50)  # 約20Hz

    #     except KeyboardInterrupt:
    #         print("finish controller")

    def get_joystick(self):
        try:
            while True:
                pygame.event.pump()

                # --- ボタン入力 ---
                for key in self.button_num.keys():
                    self.pushed_button = self.joystick.get_button(self.button_num[key])
                    self.button_state[key] = self.pushed_button
                    if self.pushed_button == 1:
                        print(f"[ボタン{key}] 押下中")

                r2_value = self.joystick.get_axis(5)
                r2_normalized = (r2_value + 1) / 2 # 0.0〜1.0 に正規化

                self.button_state["gripper"] = r2_normalized

                # if l2_normalized > 0.05:
                #     print(f"L2 押し込み量: {l2_normalized:.2f}")
                # if r2_normalized > 0.05:
                #     print(f"R2 押し込み量: {r2_normalized:.2f}")
                print(self.button_state)
                
                pygame.time.wait(50)  # 約20Hz

        except KeyboardInterrupt:
            print("finish controller")

    def get_button_state(self):
        return self.button_state

if __name__ == "__main__":
    controller = Controller()
    controller.get_joystick()
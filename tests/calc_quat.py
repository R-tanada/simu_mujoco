import numpy as np

def euler_to_quat_wxyz(roll, pitch, yaw):
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

def quat_to_euler_wxyz(quat):
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

# 例
quat = np.array([1, 1, 0, 0])  # [w, x, y, z]
roll, pitch, yaw = quat_to_euler_wxyz(quat)
print("Euler angles (rad) - roll, pitch, yaw:", roll, pitch, yaw)


# 例
roll = np.pi/2
pitch = 0
yaw = 0

quat = euler_to_quat_wxyz(roll, pitch, yaw)
print("Quaternion [w, x, y, z]:", quat)

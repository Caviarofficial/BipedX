import mujoco
import mujoco.viewer

# 模型路径（使用改进版）
xml_path = "/home/caviar/mujoco/simple_car_improved.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 基础角速度（rad/s），正值为前进方向
base_speed = 10.0

def key_callback(keycode):
    """方向键控制小车，空格停止"""
    if keycode == 32:  # 空格
        data.ctrl[:] = 0.0
        print("停止")
        return

    # 方向键码：上=265，下=264，左=263，右=262
    if keycode == 265:  # 上：前进
        data.ctrl[:] = base_speed
        print("前进")
    elif keycode == 264:  # 下：后退
        data.ctrl[:] = -base_speed
        print("后退")
    elif keycode == 263:  # 左：左转（左轮反转，右轮正转）
        data.ctrl[0] = -base_speed   # 左前
        data.ctrl[1] =  base_speed   # 右前
        data.ctrl[2] = -base_speed   # 左后
        data.ctrl[3] =  base_speed   # 右后
        print("左转")
    elif keycode == 262:  # 右：右转（左轮正转，右轮反转）
        data.ctrl[0] =  base_speed
        data.ctrl[1] = -base_speed
        data.ctrl[2] =  base_speed
        data.ctrl[3] = -base_speed
        print("右转")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0, 0, 0.1]

    print("控制说明：")
    print("  ↑ - 前进")
    print("  ↓ - 后退")
    print("  ← - 左转")
    print("  → - 右转")
    print("  空格 - 停止")
    print("关闭窗口退出仿真")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
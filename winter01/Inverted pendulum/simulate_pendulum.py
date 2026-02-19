import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
xml_path = "inverted_pendulum.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# PID 参数（需根据实际整定）
Kp = 100.0   # 比例系数
Ki = 10.0    # 积分系数
Kd = 20.0    # 微分系数

# 状态变量
integral_error = 0.0
prev_angle_error = 0.0

# 设置初始状态：摆杆稍微偏离直立（例如 0.1 弧度）
data.qpos[1] = 0.1  # hinge 关节的角度（弧度）
data.qvel[1] = 0.0

# 仿真参数
sim_time = 0.0
dt = model.opt.timestep

# 启动可视化 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 获取当前状态
        cart_pos = data.qpos[0]          # 小车位置（不重要）
        pole_angle = data.qpos[1]        # 摆杆角度（0 = 竖直向上）
        pole_vel = data.qvel[1]           # 摆杆角速度

        # 计算角度误差（目标角度为 0）
        angle_error = -pole_angle         # 误差 = 目标 - 当前

        # PID 控制律
        integral_error += angle_error * dt
        derivative = (angle_error - prev_angle_error) / dt
        control_force = Kp * angle_error + Ki * integral_error + Kd * derivative

        # 限制控制力（执行器范围已在 xml 中设置，但可以再限制一下）
        max_force = 100.0
        control_force = np.clip(control_force, -max_force, max_force)

        # 施加控制力到执行器
        data.ctrl[0] = control_force

        # 记录误差用于下一时刻微分
        prev_angle_error = angle_error

        # 前进一步仿真
        mujoco.mj_step(model, data)

        # 更新 viewer（自动刷新）
        viewer.sync()

        # 控制仿真速度（可选）
        time.sleep(dt / 2)  # 可调节以匹配实时
import mujoco
import mujoco.viewer

xml_path = "/home/caviar/mujoco/double_pendulum.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"几何体数量: {model.ngeom}")  # 应该输出 3

# 设置初始关节角度
data.qpos[0] = 0.5
data.qpos[1] = 0.3

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0, 0, 0]
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
import mujoco
import mujoco.viewer

xml_path = "/home/caviar/mujoco/simple_quadruped.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"几何体数量: {model.ngeom}")
print(f"关节数量: {model.njnt}")
print(f"执行器数量: {model.nu}")

# 设置初始关节角度（让四足动物稍微站起来）
# 假设所有膝关节弯曲 -0.5 弧度，髋关节保持中立
for i in range(4):
    data.qpos[2*i + 1] = -0.5  # 膝关节索引：奇数位置（假设排序为[hip, knee]）

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.lookat = [0, 0, 0.2]

    while viewer.is_running():
        # 可以在这里添加控制，例如让所有膝关节缓慢摆动
        # data.ctrl[4] = 0.5  # 示例
        mujoco.mj_step(model, data)
        viewer.sync()
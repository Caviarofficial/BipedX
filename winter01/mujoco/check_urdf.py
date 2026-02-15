import mujoco

try:
    model = mujoco.MjModel.from_xml_path("/home/caviar/mujoco/double_pendulum.urdf")
    print("模型加载成功")
    print("几何体数量:", model.ngeom)
except Exception as e:
    print("加载失败，错误信息：", e)
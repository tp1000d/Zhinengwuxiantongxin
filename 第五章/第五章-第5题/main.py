# 第 5 题：把 munich 场景中 "Altes_Rathaus-itu_marble" 的材质由大理石改为混凝土，并打印材质属性
# 适用版本：sionna-rt >= 1.0（本文用 sionna-rt 2.2.0 实测）
import sionna.rt
from sionna.rt import load_scene, ITURadioMaterial

# 注意：必须 merge_shapes=False，否则同材质建筑会被合并，按名字取不到 Altes_Rathaus
scene = load_scene(sionna.rt.scene.munich, merge_shapes=False)


def print_material(mat, title):
    print(f"--- {title} ---")
    print("材质名称 name                 :", mat.name)
    print("ITU 材料类型 itu_type          :", mat.itu_type)
    print("相对介电常数 relative_permittivity:", mat.relative_permittivity)
    print("电导率 conductivity [S/m]      :", mat.conductivity)
    print("厚度 thickness [m]             :", mat.thickness)
    print("散射系数 scattering_coefficient :", mat.scattering_coefficient)
    print("交叉极化系数 xpd_coefficient    :", mat.xpd_coefficient)
    print("散射方向图 scattering_pattern   :", type(mat.scattering_pattern).__name__)
    print("显示颜色 color                 :", mat.color)


obj = scene.get("Altes_Rathaus-itu_marble")
print("对象名:", obj.name, "| 当前载波频率:", scene.frequency, "Hz")
print_material(obj.radio_material, "修改前")

# 方法一：新建一个名为 "itu_concrete" 的 ITU 混凝土材质并赋给该建筑（名字与题目一致）
itu_concrete = ITURadioMaterial(name="itu_concrete", itu_type="concrete", thickness=0.1)
obj.radio_material = itu_concrete
# 方法二（等价）：直接复用场景里已有的混凝土材质
# obj.radio_material = scene.get("concrete")

print_material(obj.radio_material, "修改后")

# 补充：ITU 材料的电导率随频率变化，改变 scene.frequency 后会自动更新
scene.frequency = 28e9
print_material(obj.radio_material, "修改后（载波 28 GHz）")

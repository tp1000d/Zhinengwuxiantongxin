# 第 4 题：Sionna RT 入门教程（Introduction to Sionna RT）中的 munich 场景
# 生成：① 场景渲染图 ② 射线（传播路径）图 ③ 信道冲激响应 ④ 强度覆盖图（radio map）
# 适用版本：sionna-rt >= 1.0（本文用 sionna-rt 2.2.0 实测）
import matplotlib
matplotlib.use("Agg")              # 在 jupyter 中可删去此两行，改用 %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mitsuba as mi
import sionna.rt
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       Camera, PathSolver, RadioMapSolver)

OUT = Path(__file__).resolve().parent / "results"   # 结果保存到本题目录下的 results/
OUT.mkdir(exist_ok=True)

print("Mitsuba variant:", mi.variant())   # CPU 上应为 llvm_ad_mono_polarized

# 1) 加载内置 munich 场景（慕尼黑老城区，merge_shapes=True 合并同材质物体以加速）
scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
print("载波频率 f_c =", scene.frequency, "Hz")   # 默认 3.5 GHz

# 2) 配置收发天线阵列（此处与教程一致：各 1 个阵元）
scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                             vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                             vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="dipole", polarization="cross")

# 3) 放置发射机（基站）和接收机（用户），并让发射机朝向接收机
tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=2)
rx = Receiver(name="rx", position=[45, 90, 1.5], display_radius=2)
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

# 4) 渲染场景（无 GUI 时用 render_to_file；jupyter 中可用 scene.preview()）
my_cam = Camera(position=[-250, 250, 150], look_at=[-15, 30, 28])
scene.render_to_file(camera=my_cam, filename=str(OUT / "munich_scene.png"),
                     resolution=[650, 500], num_samples=128)

# 5) 射线追踪：直射 + 镜面反射 + 折射，最大反射深度 5
p_solver = PathSolver()
paths = p_solver(scene=scene, max_depth=5, los=True,
                 specular_reflection=True, diffuse_reflection=False,
                 refraction=True, synthetic_array=False, seed=41)
scene.render_to_file(camera=my_cam, filename=str(OUT / "munich_rays.png"), paths=paths,
                     clip_at=20, resolution=[650, 500], num_samples=128)

# 6) 由路径得到信道冲激响应 (CIR)
a, tau = paths.cir(normalize_delays=True, out_type="numpy")
print("a 的形状:", a.shape)     # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
print("tau 的形状:", tau.shape)
t = tau[0, 0, 0, 0, :] / 1e-9
a_abs = np.abs(a)[0, 0, 0, 0, :, 0]
valid = a_abs > 0
print("有效路径数:", int(valid.sum()))
plt.figure()
plt.stem(t[valid], a_abs[valid])
plt.title("Channel impulse response (munich)")
plt.xlabel(r"$\tau$ [ns]"); plt.ylabel(r"$|a|$")
plt.savefig(str(OUT / "munich_cir.png"), dpi=150, bbox_inches="tight")

# 7) 强度覆盖图（radio map），网格 1 m × 1 m
rm_solver = RadioMapSolver()
rm = rm_solver(scene=scene, max_depth=5, cell_size=[1, 1], samples_per_tx=10**6)

# 7a) 3D 视角：覆盖图叠加在场景上
scene.render_to_file(camera=my_cam, filename=str(OUT / "munich_radiomap_3d.png"),
                     radio_map=rm, rm_metric="path_gain", rm_vmin=-150,
                     resolution=[650, 500], num_samples=128)
# 7b) 2D 俯视图：路径增益 (dB)
fig = rm.show(metric="path_gain", vmin=-150)
plt.xlim(550, 1100); plt.ylim(550, 1000)   # 放大到发射机附近，便于观察
plt.savefig(str(OUT / "munich_radiomap_2d.png"), dpi=150, bbox_inches="tight")
print("完成：已生成 munich_scene/rays/cir/radiomap_3d/radiomap_2d 五张图")

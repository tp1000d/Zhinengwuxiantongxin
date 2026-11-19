# 第 3 题：DeepMIMO O1 场景数据集生成（deepmimo v4）
# 安装：pip install deepmimo
import numpy as np
import matplotlib.pyplot as plt
import deepmimo as dm

scen = "o1_28"                  # O1 场景 28 GHz 版本，也可换成 o1_3p5、o1_60
dm.download(scen)               # 从 deepmimo.net 下载射线追踪数据（只需一次）
dm.summary(scen)                # 打印场景概要：载频、基站数、用户网格、射线追踪设置

# O1 一共 18 个基站 × 118 万用户，全部载入会占用很大内存，建议只载入需要的基站。
# 先查看 TX/RX 集合的编号，再按实际打印结果修改下面的 BS_SET_ID / BS_IDX。
print(dm.get_txrx_sets(scen))
BS_SET_ID, BS_IDX = 1, 2        # 示例：BS3 = 基站集合 1 中的第 3 个（索引从 0 开始）
dataset = dm.load(scen, tx_sets={BS_SET_ID: [BS_IDX]},
                  max_paths=10, compat_v3=True)   # compat_v3：按 v3 的行/列编号合并用户网格

# 选择用户：R1000~R1100（行号从 0 开始，所以是 999~1099）
rows = np.arange(999, 1100)
idxs = dataset.get_idxs("row", row_idxs=rows)
ds = dataset.trim(idxs=idxs)
print("选中的用户数:", ds.n_ue)                  # 预期 101 × 181 = 18281
print("基站位置 (x, y, z):", ds.tx_pos)

# 信道参数：BS 为 32 元 ULA（λ/2 间距），UE 单天线；OFDM 512 个子载波中取前 64 个，带宽 100 MHz
ch_params = dm.ChannelParameters(
    bs_antenna={"shape": [32, 1], "spacing": 0.5},
    ue_antenna={"shape": [1, 1]},
    ofdm={"subcarriers": 512, "selected_subcarriers": np.arange(64), "bandwidth": 100e6},
    num_paths=10,
)
H = ds.compute_channels(ch_params)               # [n_ue, n_rx_ant, n_tx_ant, n_subcarriers]
print("信道张量 H 的形状:", H.shape)              # 预期 (18281, 1, 32, 64)

# (1) 用户与基站位置分布，按路径损耗着色
plt.figure(figsize=(8, 4))
sc = plt.scatter(ds.rx_pos[:, 0], ds.rx_pos[:, 1], c=ds.pathloss, s=1, cmap="viridis")
plt.scatter(ds.tx_pos[..., 0], ds.tx_pos[..., 1], marker="^", s=120, c="r", label="BS3")
plt.colorbar(sc, label="Path loss [dB]")
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.legend(); plt.title("O1: BS3, rows R1000-R1100")
plt.savefig("o1_users.png", dpi=150, bbox_inches="tight")

# (2) 选 3 个点：R1000 行的第 1、91、181 个用户
picks = [0, 90, 180]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, u in zip(axs, picks):
    im = ax.imshow(np.abs(H[u, 0]), aspect="auto", cmap="viridis")  # 天线 × 子载波
    ax.set_title(f"User {u}  pos={np.round(ds.rx_pos[u, :2], 1)}  LoS={ds.los[u]}")
    ax.set_xlabel("Subcarrier index"); ax.set_ylabel("BS antenna index")
    fig.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig("o1_channels.png", dpi=150)

# (3) 打印信道矩阵（前 4 根天线 × 前 4 个子载波）
np.set_printoptions(precision=3, suppress=False, linewidth=120)
for u in picks:
    print(f"\n用户 {u} 的信道矩阵 H[:4, :4]（行：BS 天线，列：子载波）:")
    print(H[u, 0, :4, :4])

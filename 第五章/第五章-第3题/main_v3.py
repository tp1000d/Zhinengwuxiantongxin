# pip install "deepmimo==3.2.9"  （模块名为 DeepMIMOv3；需下载 v3 格式的 O1 场景文件）
import numpy as np
import DeepMIMOv3 as DeepMIMO

params = DeepMIMO.default_params()
params["dataset_folder"] = "./scenarios"        # 解压后的场景文件夹
params["scenario"] = "O1_28"
params["num_paths"] = 10
params["active_BS"] = np.array([3])             # 基站编号从 1 开始
params["user_rows"] = np.arange(999, 1100)      # R1000~R1100（行号从 0 开始）
params["user_subsampling"] = 1                  # 1 表示该行用户全部保留
params["bs_antenna"]["shape"] = np.array([32, 1])
params["bs_antenna"]["spacing"] = 0.5
params["ue_antenna"]["shape"] = np.array([1, 1])
params["enable_BS2BS"] = 0
params["OFDM_channels"] = 1                     # 1：OFDM 频域信道；0：时域
params["OFDM"]["subcarriers"] = 512
params["OFDM"]["selected_subcarriers"] = np.arange(64)
params["OFDM"]["bandwidth"] = 0.1               # 注意 v3 的单位是 GHz

dataset = DeepMIMO.generate_data(params)
H = dataset[0]["user"]["channel"]               # [用户数, UE 天线, BS 天线, 子载波]
loc = dataset[0]["user"]["location"]            # 用户坐标
los = dataset[0]["user"]["LoS"]                 # 1: LoS, 0: NLoS, -1: 无路径
bs_loc = dataset[0]["location"]                 # 基站坐标
print(H.shape, bs_loc)

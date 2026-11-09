# 第三章三道习题参考答案

按 `problems.md` 的实际题号以及确认后的编号，分别使用以下目录。

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| [第三章-第12题](第三章-第12题/README.md) | PCA：协方差、特征分解、降维和重构 | `main.py` |
| [第三章-第13题](第三章-第13题/README.md) | 手工反向传播、全部权重和偏置的梯度 | `main.py` |
| [第三章-第14题](第三章-第14题/README.md) | CNN 与 LS 的上行 MIMO 信道估计比较 | `main.py`、`verify.py` |

## 运行

在本目录执行，三个实验都使用 `pt2` 环境：

```bash
conda activate pt2
python 第三章-第12题/main.py
python 第三章-第13题/main.py
python 第三章-第14题/verify.py
python 第三章-第14题/main.py
```

也可以不激活环境，使用 `conda run --no-capture-output -n pt2 python` 替换上述命令中的 `python`。

依赖为 NumPy、PyTorch、Matplotlib，当前 `pt2` 均已具备，无须重新安装。实验实际使用 Python 3.9.18、NumPy 1.23.2、PyTorch 2.1.0、Matplotlib 3.8.4；CNN 在 NVIDIA RTX 5000 Ada Generation 上训练。各程序默认将结果写到本题目录下的 `results/`，重复运行会更新这些结果。

## 已完成的验证

- 第12题：协方差特征分解与独立 SVD 结果一致，并通过正交性、特征方程、重构与误差校验；前两个主成分累计解释方差为 **99.6087%**。
- 第13题：9 个参数的手工梯度均与中心差分、PyTorch 自动微分一致；中心差分最大绝对误差为 **6.13 × 10⁻¹² 以下**。题目缺少激活函数，答案明确采用两层 Sigmoid 和 `L=(y-t)²`。
- 第14题：阵列模型、复噪声归一化、LS 理论 NMSE、网络形状和梯度检查通过；默认 25 轮训练后，独立测试集 LS 为 **−10.0467 dB**、CNN 为 **−14.5303 dB**，改善 **4.4837 dB**。模型保存后重新加载的输出也已核验。

具体公式、数值答案、运行选项和结果文件说明见各题的 `README.md`。数值以随附的 `results/results.json` 为准。

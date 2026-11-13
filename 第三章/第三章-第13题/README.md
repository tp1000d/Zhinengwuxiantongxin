# 第三章 第13题：反向传播算法

## 必须说明的假设

原题未注明激活函数，因此题面无法确定唯一的数值梯度。本参考答案采用常见设定：隐藏层和输出层都使用 Sigmoid，$\sigma(z)=1/(1+e^{-z})$。针对一个样本、一个输出，MSE 定义为：

$$
L=(y-t)^2,\qquad t=0.6.
$$

这里没有额外乘 $1/2$。若教材使用 $L=\tfrac12(y-t)^2$，下文所有梯度都应除以 2；若激活函数不同，须重新进行前向和反向计算。

## 前向传播

严格按题目中 $w_{ij}$ 表示 $x_i\to h_j$ 的约定：

$$
\begin{aligned}
z_1&=0.5\times0.2+0.1\times0.1+0.1=0.21,\\
z_2&=0.5\times0.4+0.1\times0.3+0.2=0.43,\\
h_1&=\sigma(z_1)=0.552307909574,\\
h_2&=\sigma(z_2)=0.605873668432,\\
z_y&=0.3h_1+0.5h_2+0.1=0.568629207088,\\
y&=\sigma(z_y)=0.638446811711,\\
L&=(y-0.6)^2=0.001478157331.
\end{aligned}
$$

## 反向传播

由于 $\sigma'(z)=\sigma(z)(1-\sigma(z))$，输出层误差信号为：

$$
\delta_y=\frac{\partial L}{\partial z_y}
=2(y-t)y(1-y)=0.017749545816.
$$

隐藏层误差信号为：

$$
\begin{aligned}
\delta_{h_1}&=\delta_y v_1 h_1(1-h_1)=0.001316646484,\\
\delta_{h_2}&=\delta_y v_2 h_2(1-h_2)=0.002119213824.
\end{aligned}
$$

由链式法则：

$$
\frac{\partial L}{\partial v_j}=\delta_y h_j,\quad
\frac{\partial L}{\partial b_y}=\delta_y,\quad
\frac{\partial L}{\partial w_{ij}}=x_i\delta_{h_j},\quad
\frac{\partial L}{\partial b_{h_j}}=\delta_{h_j}.
$$

全部权重和偏置的梯度如下：

| 参数 | 梯度 |
| --- | ---: |
| $w_{11}$ | 0.000658323242 |
| $w_{12}$ | 0.001059606912 |
| $w_{21}$ | 0.000131664648 |
| $w_{22}$ | 0.000211921382 |
| $b_{h_1}$ | 0.001316646484 |
| $b_{h_2}$ | 0.002119213824 |
| $v_1$ | 0.009803214545 |
| $v_2$ | 0.010753982436 |
| $b_y$ | 0.017749545816 |

## 运行说明

在本题目录运行：

```bash
conda activate pt2
python main.py
```

或在上一级目录运行 `conda run --no-capture-output -n pt2 python 第三章-第13题/main.py`。

依赖 NumPy、PyTorch，CPU 即可完成。`main.py` 的 `forward()` 进行前向传播，`backward()` 显式实现链式法则；`finite_difference()` 与 `autograd()` 仅用于核验答案。

程序打印中间值以及手工、中心差分、自动微分三组梯度，保存完整精度数据至 `results/results.json`。

## 实际验证

已在 `pt2` 中运行，使用双精度完成以下检查：

- 对全部 9 个参数，用 $\varepsilon=10^{-6}$ 的中心差分 $[L(p+\varepsilon)-L(p-\varepsilon)]/(2\varepsilon)$ 核验；最大绝对误差为 **6.1221 × 10⁻¹²**。
- 与 PyTorch 自动微分逐项比较，最大绝对误差为 **3.4695 × 10⁻¹⁸**。
- 以学习率 0.1 沿负梯度更新一次参数，损失从 **0.001478157331** 降至 **0.001425156734**。

上述检查均通过断言；手工推导结果作为最终参考梯度。

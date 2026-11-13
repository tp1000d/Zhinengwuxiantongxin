"""沿天线维度做卷积，估计 LS 结果中应被去掉的噪声。"""

from torch import nn


class ChannelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_estimator = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 2, kernel_size=9, padding=4),
        )
        # 初始输出恰为 LS，便于观察训练带来的改善。
        nn.init.zeros_(self.noise_estimator[-1].weight)
        nn.init.zeros_(self.noise_estimator[-1].bias)

    def forward(self, received):
        return received - self.noise_estimator(received)

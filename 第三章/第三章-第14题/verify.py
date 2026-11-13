"""检查信道/噪声归一化、LS 理论值以及 CNN 输入输出约定。"""

import numpy as np
import torch

from data import as_real_tensor, simulate, steering_vector
from model import ChannelCNN


def main():
    torch.set_num_threads(4)
    # theta=0 时所有天线同相；theta=pi/6 时相邻天线相位差为 pi/2。
    responses = steering_vector(np.array([0.0, np.pi / 6]))
    np.testing.assert_allclose(responses[0], np.ones(64), atol=1e-12)
    np.testing.assert_allclose(responses[1, 1:] / responses[1, :-1], 1j, atol=1e-12)
    np.testing.assert_allclose(np.abs(responses) ** 2, np.ones((2, 64)), atol=1e-12)

    channel, received = simulate(10000, snr_db=10.0, seed=2026)
    repeated_channel, repeated_received = simulate(10000, snr_db=10.0, seed=2026)
    np.testing.assert_array_equal(channel, repeated_channel)
    np.testing.assert_array_equal(received, repeated_received)
    noise = received - channel
    channel_power = float(np.mean(np.abs(channel) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    ls_nmse_db = float(10 * np.log10(noise_power / channel_power))
    assert abs(channel_power - 1.0) < 0.04
    assert abs(noise_power - 0.1) < 0.002
    assert abs(float(noise.real.var()) - 0.05) < 0.002
    assert abs(float(noise.imag.var()) - 0.05) < 0.002
    assert abs(ls_nmse_db - (-10.0)) < 0.25

    tensor = as_real_tensor(received[:8])
    assert tensor.shape == (8, 2, 64) and tensor.dtype == torch.float32
    np.testing.assert_array_equal(tensor[:, 0].numpy() + 1j * tensor[:, 1].numpy(), received[:8])
    model = ChannelCNN()
    estimate = model(tensor)
    torch.testing.assert_close(estimate, tensor, rtol=0, atol=0)
    loss = (estimate - as_real_tensor(channel[:8])).square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    print(f"信道平均功率: {channel_power:.6f}（理论 1）")
    print(f"复噪声平均功率: {noise_power:.6f}（理论 0.1）")
    print(f"LS NMSE: {ls_nmse_db:.4f} dB（理论约 -10 dB）")
    print("验证通过：阵列相位、随机复现、功率归一化、LS 理论值、数据形状和梯度。")


if __name__ == "__main__":
    main()

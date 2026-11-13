"""单天线用户到 64 天线 ULA 的窄带上行信道与单位导频观测。"""

import numpy as np
import torch


def complex_normal(rng, shape):
    """CN(0, 1)：实部、虚部的方差各为 1/2。"""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def steering_vector(angles, antennas=64):
    """半波长间距 ULA；每个阵列响应的模平方之和为 antennas。"""
    indices = np.arange(antennas)
    return np.exp(1j * np.pi * np.sin(angles)[..., None] * indices)


def simulate(samples, snr_db=10.0, seed=42, antennas=64, paths=3):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi / 2, np.pi / 2, size=(samples, paths))
    gains = complex_normal(rng, (samples, paths))
    channel = (gains[..., None] * steering_vector(angles, antennas)).sum(axis=1) / np.sqrt(paths)
    # E|h_m|^2 = 1，单位导频 x = 1，因此 E|n_m|^2 = 10^(-SNR/10)。
    noise = np.sqrt(10 ** (-snr_db / 10)) * complex_normal(rng, channel.shape)
    received = channel + noise
    return channel.astype(np.complex64), received.astype(np.complex64)


def as_real_tensor(values):
    """(样本数, 天线数) 的复数数组 -> (样本数, 2, 天线数) float32。"""
    return torch.from_numpy(np.stack([values.real, values.imag], axis=1).astype(np.float32))


def make_dataset(samples, snr_db, seed):
    channel, received = simulate(samples, snr_db, seed)
    return torch.utils.data.TensorDataset(as_real_tensor(received), as_real_tensor(channel))

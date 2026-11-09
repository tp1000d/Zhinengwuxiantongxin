"""在独立仿真数据上训练 CNN，并在未参与训练/选模的测试集与 LS 比较。"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import make_dataset
from model import ChannelCNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate(model, loader, device):
    """先对所有样本累加能量，再计算 NMSE，避免对 dB 值取平均。"""
    model.eval()
    cnn_error, ls_error, energy = 0.0, 0.0, 0.0
    with torch.no_grad():
        for received, target in loader:
            received, target = received.to(device), target.to(device)
            estimate = model(received)
            cnn_error += (estimate - target).double().square().sum().item()
            ls_error += (received - target).double().square().sum().item()
            energy += target.double().square().sum().item()
    return {"cnn_nmse": cnn_error / energy, "ls_nmse": ls_error / energy}


def train(model, train_loader, val_loader, device, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    best_nmse, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for received, target in train_loader:
            received, target = received.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(received), target)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现 NaN 或 Inf。")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
        metrics = evaluate(model, val_loader, device)
        val_nmse = metrics["cnn_nmse"]
        if val_nmse < best_nmse:
            best_nmse, best_epoch = val_nmse, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_mse": loss_sum / len(train_loader.dataset),
                        "val_nmse": val_nmse, "val_nmse_db": 10 * np.log10(val_nmse)})
        scheduler.step()
        print(f"Epoch {epoch:2d}/{epochs}: train MSE={history[-1]['train_mse']:.6f}, "
              f"validation NMSE={history[-1]['val_nmse_db']:.3f} dB", flush=True)
    model.load_state_dict(best_state)
    return history, best_epoch


def save_plots(history, metrics, example, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["train_mse"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Training MSE")
    axes[1].plot([r["epoch"] for r in history], [r["val_nmse_db"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation NMSE (dB)")
    fig.tight_layout()
    fig.savefig(output_dir / "training.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["LS", "CNN"], [metrics["ls_nmse_db"], metrics["cnn_nmse_db"]])
    ax.bar_label(bars, fmt="%.2f dB", padding=3)
    ax.set(ylabel="Test NMSE (dB), lower is better", title="64 antennas, 3 paths, SNR = 10 dB")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for index, label in enumerate(["Real part", "Imaginary part"]):
        axes[index].plot(example["true"][index], label="True channel")
        axes[index].plot(example["ls"][index], ".", alpha=0.7, label="LS")
        axes[index].plot(example["cnn"][index], "--", label="CNN")
        axes[index].set(ylabel=label)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Antenna index")
    fig.tight_layout()
    fig.savefig(output_dir / "example.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--train-samples", type=int, default=24000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0 等")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    for name in ["epochs", "train_samples", "val_samples", "test_samples", "batch_size", "learning_rate"]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须大于 0")
    if not 0 <= args.seed < 2 ** 32 - 3:
        parser.error("seed 必须在 [0, 2^32-4] 内")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    # 三个不同随机种子，互不复用样本；测试集仅在选模结束后使用。
    train_set = make_dataset(args.train_samples, 10.0, args.seed)
    val_set = make_dataset(args.val_samples, 10.0, args.seed + 1)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = ChannelCNN().to(device)
    print(f"Python: {sys.executable}\nPyTorch: {torch.__version__}\nDevice: {device}", flush=True)
    history, best_epoch = train(model, train_loader, val_loader, device, args.epochs, args.learning_rate)

    test_set = make_dataset(args.test_samples, 10.0, args.seed + 2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    metrics = evaluate(model, test_loader, device)
    metrics["ls_nmse_db"] = float(10 * np.log10(metrics["ls_nmse"]))
    metrics["cnn_nmse_db"] = float(10 * np.log10(metrics["cnn_nmse"]))
    metrics["gain_db"] = metrics["ls_nmse_db"] - metrics["cnn_nmse_db"]
    metrics["error_reduction_percent"] = 100 * (1 - metrics["cnn_nmse"] / metrics["ls_nmse"])
    assert np.isfinite(list(metrics.values())).all(), "评价指标必须有限"
    assert metrics["cnn_nmse"] < metrics["ls_nmse"], "CNN 尚未优于 LS，请增加训练样本/轮数"

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    # 存 CPU 权重，便于在没有 GPU 的机器上加载。
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, checkpoint_path)
    reloaded = ChannelCNN().to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reloaded.eval()
    model.eval()
    received, target = test_set[0]
    with torch.no_grad():
        batch = received[None].to(device)
        estimate = model(batch)
        torch.testing.assert_close(reloaded(batch), estimate)
    example = {"true": target.numpy(), "ls": received.numpy(), "cnn": estimate[0].cpu().numpy()}
    np.savez(output_dir / "example.npz", **example)
    results = {
        "config": {**vars(args), "output": str(output_dir), "device_used": str(device),
                   "antennas": 64, "paths": 3, "snr_db": 10.0,
                   "train_seed": args.seed, "val_seed": args.seed + 1, "test_seed": args.seed + 2},
        "environment": {"python_executable": sys.executable, "python": sys.version.split()[0],
                        "numpy": np.__version__, "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "test_metrics": metrics, "history": history,
        "verification": {"cnn_beats_ls": True, "checkpoint_reload_matches": True},
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(history, metrics, example, output_dir)
    print(f"\nBest epoch: {best_epoch}\nLS test NMSE: {metrics['ls_nmse_db']:.4f} dB\n"
          f"CNN test NMSE: {metrics['cnn_nmse_db']:.4f} dB\nGain: {metrics['gain_db']:.4f} dB\n"
          f"Error reduction: {metrics['error_reduction_percent']:.2f}%\nResults: {output_dir}")
    print("验证通过：指标有限、独立测试集上 CNN 优于 LS、保存后加载的模型输出一致。")


if __name__ == "__main__":
    main()

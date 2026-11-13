"""对题目给定的四维数据做 PCA，并用 SVD 和重构验证结果。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = np.array([
    [2.5, 2.4, 3.5, 3.0],
    [0.5, 0.7, 1.2, 1.1],
    [2.2, 2.9, 3.1, 3.2],
    [1.9, 2.2, 2.5, 2.8],
], dtype=np.float64)


def pca(data, n_components=2):
    """只中心化，使用样本协方差；components 的每一列是一个主成分。"""
    mean = data.mean(axis=0)
    centered = data - mean
    covariance = centered.T @ centered / (len(data) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    # 特征向量正负号等价；统一最大绝对值元素为正，便于复现实验表格。
    for column in range(components.shape[1]):
        pivot = np.argmax(np.abs(components[:, column]))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    scores = centered @ components[:, :n_components]
    reconstructed = scores @ components[:, :n_components].T + mean
    ratios = eigenvalues / eigenvalues.sum()
    return {
        "mean": mean,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "explained_variance_ratio": ratios,
        "cumulative_variance_ratio": ratios.cumsum(),
        "scores": scores,
        "reconstructed": reconstructed,
        "reconstruction_mse": float(np.mean((data - reconstructed) ** 2)),
        "components_for_95_percent": int(np.searchsorted(ratios.cumsum(), 0.95) + 1),
    }


def verify(data, result):
    centered = data - result["mean"]
    values = result["eigenvalues"]
    vectors = result["components"]
    # SVD 是独立于协方差特征值分解的验证方法。
    singular_values = np.linalg.svd(centered, compute_uv=False)
    np.testing.assert_allclose(values, singular_values ** 2 / (len(data) - 1), atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(data.shape[1]), atol=1e-12)
    np.testing.assert_allclose(result["covariance"] @ vectors, vectors * values, atol=1e-12)
    np.testing.assert_allclose(centered @ vectors @ vectors.T + result["mean"], data, atol=1e-12)
    np.testing.assert_allclose(result["explained_variance_ratio"].sum(), 1.0)
    kept = result["scores"].shape[1]
    expected_mse = values[kept:].sum() * (len(data) - 1) / data.size
    np.testing.assert_allclose(result["reconstruction_mse"], expected_mse, atol=1e-12)
    assert np.linalg.matrix_rank(centered) == 3


def save_plot(result, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ratios = result["explained_variance_ratio"]
    axes[0].bar(np.arange(1, 5), ratios * 100)
    axes[0].plot(np.arange(1, 5), ratios.cumsum() * 100, "o-", color="tab:orange")
    axes[0].set(xlabel="Principal component", ylabel="Explained variance (%)", xticks=[1, 2, 3, 4])
    scores = result["scores"]
    axes[1].scatter(scores[:, 0], scores[:, 1])
    for i, point in enumerate(scores, start=1):
        axes[1].annotate(str(i), point, xytext=(5, 5), textcoords="offset points")
    axes[1].set(xlabel="PC1 score", ylabel="PC2 score", title="PCA projection")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "pca.png", dpi=160)
    plt.close(fig)


def main():
    np.set_printoptions(precision=9, suppress=True)
    result = pca(DATA)
    verify(DATA, result)
    for name, value in result.items():
        print(f"{name}:\n{value}\n")
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    serializable = {name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in result.items()}
    (output_dir / "results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    save_plot(result, output_dir)
    print("验证通过：SVD、特征方程、正交性、全维重构和降维误差。")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()

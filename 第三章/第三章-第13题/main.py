"""手工反向传播，并以中心差分和 PyTorch 自动微分核验全部参数梯度。"""

import json
from pathlib import Path

import numpy as np
import torch


X1, X2, TARGET = 0.5, 0.1, 0.6
PARAMETERS = {
    "w11": 0.2, "w12": 0.4, "w21": 0.1, "w22": 0.3,
    "b_h1": 0.1, "b_h2": 0.2,
    "v1": 0.3, "v2": 0.5, "b_y": 0.1,
}


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def forward(p):
    z1 = X1 * p["w11"] + X2 * p["w21"] + p["b_h1"]
    z2 = X1 * p["w12"] + X2 * p["w22"] + p["b_h2"]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    z_y = h1 * p["v1"] + h2 * p["v2"] + p["b_y"]
    y = sigmoid(z_y)
    return {"z1": z1, "z2": z2, "h1": h1, "h2": h2,
            "z_y": z_y, "y": y, "loss": (y - TARGET) ** 2}


def backward(p, state):
    h1, h2, y = state["h1"], state["h2"], state["y"]
    delta_y = 2 * (y - TARGET) * y * (1 - y)
    delta_h1 = delta_y * p["v1"] * h1 * (1 - h1)
    delta_h2 = delta_y * p["v2"] * h2 * (1 - h2)
    return {
        "w11": delta_h1 * X1, "w12": delta_h2 * X1,
        "w21": delta_h1 * X2, "w22": delta_h2 * X2,
        "b_h1": delta_h1, "b_h2": delta_h2,
        "v1": delta_y * h1, "v2": delta_y * h2, "b_y": delta_y,
    }


def finite_difference(p, epsilon=1e-6):
    gradients = {}
    for name in p:
        plus, minus = p.copy(), p.copy()
        plus[name] += epsilon
        minus[name] -= epsilon
        gradients[name] = (forward(plus)["loss"] - forward(minus)["loss"]) / (2 * epsilon)
    return gradients


def autograd(p):
    parameters = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
                  for name, value in p.items()}
    h1 = torch.sigmoid(X1 * parameters["w11"] + X2 * parameters["w21"] + parameters["b_h1"])
    h2 = torch.sigmoid(X1 * parameters["w12"] + X2 * parameters["w22"] + parameters["b_h2"])
    y = torch.sigmoid(h1 * parameters["v1"] + h2 * parameters["v2"] + parameters["b_y"])
    loss = torch.nn.functional.mse_loss(y, torch.tensor(TARGET, dtype=torch.float64))
    loss.backward()
    return {name: value.grad.item() for name, value in parameters.items()}


def main():
    state = forward(PARAMETERS)
    manual = backward(PARAMETERS, state)
    numerical = finite_difference(PARAMETERS)
    automatic = autograd(PARAMETERS)
    names = list(PARAMETERS)
    np.testing.assert_allclose([manual[n] for n in names], [numerical[n] for n in names], atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose([manual[n] for n in names], [automatic[n] for n in names], atol=1e-14, rtol=1e-12)
    # 梯度方向的另一个检查：沿负梯度走一小步，损失应下降。
    updated = {name: PARAMETERS[name] - 0.1 * manual[name] for name in names}
    new_loss = forward(updated)["loss"]
    assert new_loss < state["loss"]
    print("假设：隐藏层、输出层均使用 Sigmoid；L = (y - t)^2。")
    for name, value in state.items():
        print(f"{name:>6} = {value:.12f}")
    print(f"\n{'参数':<8} {'手工反向传播':>18} {'中心差分':>18} {'自动微分':>18}")
    for name in names:
        print(f"{name:<8} {manual[name]:18.12f} {numerical[name]:18.12f} {automatic[name]:18.12f}")
    errors = {
        "finite_difference_max_abs_error": max(abs(manual[n] - numerical[n]) for n in names),
        "autograd_max_abs_error": max(abs(manual[n] - automatic[n]) for n in names),
    }
    result = {"activation": "sigmoid in both layers", "loss_definition": "(y - t)^2",
              "parameters": PARAMETERS, "forward": state, "manual_gradients": manual,
              "finite_difference_gradients": numerical, "autograd_gradients": automatic,
              "verification": errors, "loss_after_gradient_step": new_loss}
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n验证通过：全部 9 个梯度一致；一步梯度下降后损失为 {new_loss:.12f}。")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()

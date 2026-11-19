# 第 5 章 第 7、9、10 题数值计算
import numpy as np
# 第 7 题
pl = 32.4 + 21*np.log10(100) + 20*np.log10(2)
print(f"Q7: {pl:.2f} dB")                                  # 80.42 dB
# 第 9 题（蒙特卡洛验证阴影衰落的统计特性）
S = np.random.default_rng(0).normal(0, 4, 1_000_000)
PL = 10*3*np.log10(100) + 20 + S
print(f"Q9: 均值 {PL.mean():.2f} dB, 标准差 {PL.std():.2f} dB, "
      f"P(76~84) = {np.mean(np.abs(S) <= 4):.3f}, P(72~88) = {np.mean(np.abs(S) <= 8):.3f}")
# 第 10 题
prx = 0 + (-80)
print(f"Q10: {prx} dBm = {10**(prx/10)*1e-3:.1e} W")       # -80 dBm = 1.0e-11 W

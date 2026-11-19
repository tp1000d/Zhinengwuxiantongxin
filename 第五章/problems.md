# 第五章　智能通信数据集　习题

1. 试想如果是你需要生成一个数据集，需要采用什么样的逻辑来生成？请根据本章的内容，按照场景与处理和射线追踪算法两方面，挑选室外或者室内场景，画一张完整的流程图，要求按照自己的理解尽可能抓住重要的细节，并体现在流程图当中。

2. 请简述射线追踪算法在无线通信系统仿真中的作用和优势。

3. 在 https://www.deepmimo.net 网页中，按照它的操作流程，生成 O1 场景的数据集。要求选择几个点的信道图片及矩阵，阐明该场景的天线阵列特点，基站数量及位置，如何选择的用户，以及用户的具体数量和间隔。

4. 在 https://nvlabs.github.io/sionna 网页中，按照"Quick Start"安装流程，成功运行"Tutorials-For Tracing-Introduction to Sionna RT"流程，生成 munich 这个场景的射线图和强度覆盖图。

5. 在上一题的基础之上，在 jupyter notebook 中写一个程序，修改"Altes_Rathaus-itu_marble"材质的名字，令其从"itu_marble"变成"itu_concrete"，并打印出材质的各个属性特征。

6. 以下哪个数据集主要用于毫米波通信环境的模拟？

   (A) ImageNet　(B) CIFAR-10　(C) DeepMIMO　(D) MNIST

7. 已知在城市环境中的路径损耗模型为

   $$\mathrm{PL}_{\mathrm{UMi\text{-}LoS}} = 32.4 + 21\log_{10}(d_{3D}) + 20\log_{10}(f_c)$$

   其中，$d_{3D}$ 为三维距离，m；$f_c$ 为中心频率，GHz。计算当 $d_{3D} = 100$ m，$f_c = 2$ GHz 时的路径损耗。

8. 3GPP 38.901 信道模型的有效频率范围是：

   (A) 0.5～100 MHz　(B) 0.5～100 GHz　(C) 5～100 GHz　(D) 5～10 GHz

9. 使用经典的 α-β 模型计算路径损耗，其中 α = 3，β = 20，阴影衰落 S 的标准差 σ<sub>s</sub> = 4。若 d = 100 m，计算路径损耗。

10. 给定路径增益为 PG = −80 dB，发射功率 P<sub>TX</sub> = 0 dBm。计算接收功率 P<sub>RX</sub>。

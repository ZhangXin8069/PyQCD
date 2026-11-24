# E32P29 四点函数的已验证取舍

本例计算 $\langle O_p J_\mu^L J_\mu^L O_{n\pi^-}^\dagger\rangle$，其中 $J_\mu^L=\bar u\gamma_\mu(1-\gamma_5)d/2$。源、汇各 100 个精确模，每个流 400 个存储模。36 张图各自保留同一个 $\mu$ 和一对 $(\mathbf q,-\mathbf q)$，27 个物理动量可分成三个九动量批。冻结的是该测量的算符与有序 Wick 等式。换图、换基底或换估计量时，从新的方程推导。[生产说明](/public/home/sush/distillation/test/4pt_acc/README.md) · [组态入口](/public/home/sush/distillation/test/4pt_acc/e32p29.py)

## 降低维数前先确定输出

实际 DeGrand–Rossi 基的 $\gamma_4=\begin{pmatrix}0&I\\I&0\end{pmatrix}$。双侧投影 $P_+=(1+\gamma_4)/2$ 后，用户要的是左上块。令 $F=(I\ I)/2$，则

$$
H=FCF^\dagger=(C_{00}+C_{01}+C_{10}+C_{11})/4,\qquad
P_+CP_+=\begin{pmatrix}H&H\\H&H\end{pmatrix}.
$$

在两条携带开放旋量指标的传播子上分别吸收 $F$ 和 $F^\dagger$。其余内部旋量仍完整求和。归一基 $Q=(I,I)^T/\sqrt2$ 给出 $Q^\dagger C Q=2H$。验收时构造仅下分量非零的输入，可以检出误截原始左上块。仅有一侧投影时，需从 $P_+C$ 或 $CP_+$ 的实际块结构重算可压缩轴。[投影及归一化核查](/public/home/sush/.log_codex/2026-11-19-4pt-cuda-acceleration/projection-validation.md)

## 把共用的传播线变成 CUDA 操作

按味和端口枚举该问题的 $3!\times3!$ 个配对，从参考 Wick 结果核对每张图的费米系数。先形成流插入的顺序传播线，沿源介子或闭环组合，再对剩余的重子块做子集动态规划。用输入槽和端口配对识别跨图相同的 GEMM 和重排。分别标注节点只依赖动量、只依赖流分量、依赖两者或两者皆无，按实际通道数计代价。这里启用新投影并重新选树后，主复乘加量由 $4.033\times10^{12}$ 变为 $2.339\times10^{12}$。这个变化由各节点的尺寸算得。[路径生成器](/public/home/sush/distillation/test/4pt_acc/plan.py) · [CUDA 调度与内核](/public/home/sush/distillation/test/4pt_acc/fourpt.cu)

大矩阵留给 cuBLAS，并单独实测过重排时生成实数平面、三实乘法、长 $K$ 小输出的专用归约以及增大动量批宽。有些大块预计算消耗十余 GiB，实测时间仍更长。Python 入口按物理顺序传传播子、gamma、顶角，用 19 个张量表示本例两个相同汇端 $u$ 位置的数据。新输入可覆盖旧缓冲区，计划复用调度和工作区。[Python 接口](/public/home/sush/distillation/test/4pt_acc/fourpt.py) · [同卡对照与算力分析](/public/home/sush/.log_codex/2026-11-19-4pt-cuda-acceleration/a100-profile.md)

## 抽样改变了能用的因式分解

在参考的异时独立抽样约定下，两个流顶角分别乘自己的 $\Omega_2$。等时使用同一份样本，权重 $\Omega_4(i,j,k,l)$ 同时依赖四个本征矢指标。对本例各子空间内的不放回抽样，

$$
\Omega_4=\prod_g\frac{(N_g)_{r_g}}{(s_g)_{r_g}},
\qquad (n)_r=n(n-1)\cdots(n-r+1),
$$

其中 $r_g$ 是出现在子空间 $g$ 中的不同抽样编号数，$N_g,s_g$ 分别是总体和保留数。重复编号只计一次。生产等时分支将原始顶角交给联合四阶权重收缩。验证时在含精确、压缩和噪声模的真实子空间上独立生成稠密四阶张量，和完整旋量的 Wick 方程一起收缩。对于其他采样器，先从其联合包含概率确定权重阶数及能否分解。

## 计时所代表的工作

E32P29 组态 1000 的一个异时时间对在单张 A100 PCIe 40 GB 上，每九动量的带符号原生收缩约 1.48 s，全部 27 动量收缩阶段 4.55 s。相对从磁盘独立计算的 `lqcddb_update.dynamic_contraction` 双精度总和，复双精度输入相对 $L_2$ 差为 $2.30\times10^{-14}$，复单精度输入为 $6.36\times10^{-8}$。此带符号模式在内部使用复双精度。

同组态等时九动量块采用联合权重，含首次动态路径搜索为 922.67 s。四对相反动量的交换对称性和独立的稠密权重检查分别检验了图求和对称性与抽样归一化。以上时间限于所列设备、分支和测量边界。工作目录变化时可按 `plan.py`、`fourpt.cu` 和 `e32p29.py` 的职责定位同类代码。[真实数据的参数、误差及计时记录](/public/home/sush/.log_codex/2026-11-19-4pt-cuda-acceleration/E32P29-production.md)

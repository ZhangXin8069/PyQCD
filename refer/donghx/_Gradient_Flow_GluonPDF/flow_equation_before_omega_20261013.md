目前 quark PDF 的 Blending 框架实现的是：四维 Wilson gauge flow 与四维 fermion flow 同步演化，底层使用耦合三阶 Runge–Kutta（RK3）算法。 Python 负责准备规范场、初始 basis 和参数，实际流积分由 PyQUDA 调用 QUDA 完成。

它不是先把规范场流到终点，再在终点规范场上对 eigenvector 做一次 smearing；也不是重新求解 flowed gauge 上的 Laplacian 本征向量。

## 1. 实际求解的方程

为避免混淆，下面用 U_{μ}(t_{f},x) 表示 flowed gauge links，用 \(V_n(\boldsymbol x;\sigma)\) 表示原始基底；σ 是欧氏源时间片，而不是 flow time。

连续形式采用不加 flow-time gauge-damping 项的约定：

\[
\begin{aligned}
\partial_{t_f} B_\mu
&=D_\nu G_{\nu\mu},
& B_\mu(0)&=A_\mu,\\
\partial_{t_f}\chi
&=\Delta_{B(t_f)}\chi,
& \chi(0)&=\psi,\\
\partial_{t_f}\bar\chi
&=\bar\chi\overleftarrow{\Delta}_{B(t_f)},
&&
\end{aligned}
\qquad
\Delta_B=\sum_{\mu=1}^{4}D_\mu D_\mu.
\]

规范场从未做 HYP smear 的 thin links 开始。规范场流决定 B(t_{f})，费米子流使用演化过程中的 B(t_{f})；费米子不会反过来影响规范场流。当前 Note 的定义见 gauge/fermion flow 方程 (/public/home/donghx/Note/Gradient_Flow/main.tex:217)。

### 格点上的费米子流

代码使用最近邻四维协变 Laplacian：

\[
\begin{aligned}
(a^2\Delta_U\phi)(x)
=\sum_{\mu=1}^{4}\Big[
&U_\mu(x)\phi(x+\hat\mu)\\
+&U_\mu^\dagger(x-\hat\mu)\phi(x-\hat\mu)
-2\phi(x)\Big].
\end{aligned}
\]

也就是四个方向的正、反向跳跃之和，再减去 8ϕ(x)。

底层调用确实是：

double a = 1.;
double b = -8.;
ApplyLaplace(..., precise, 4, a, b, ...);

这里 4 表示包含时间方向的四维 Laplacian；源码见 QUDA 的流算符参数 (/public/home/donghx/PyQuda_a100_2608/quda/lib/interface_quda.cpp:5543)。它没有 Dirac 的质量项、gamma hopping 或 Clover 项，不是在做 Dirac 求逆。

## 2. 一个 flow step 怎样更新两个场？

定义无量纲流时间

  t_{f}
τ=─────,   ϵ=Δτ,   L[U]=a²Δ_{U}.
   a²

设一步开始时的场为 U₀,ϕ₀，并用 𝒵[U] 表示代码约定下的 Wilson gauge-force 生成元。下面的 𝒵 已包含相应符号约定。

第一阶段：

k₀ =L[U₀]ϕ₀,
       ϵ
ϕ₁ =ϕ₀+─k₀,
       4
        ϵ
U₁ =exp[─𝒵[U₀]]U₀.
        4

第二阶段：

k₁ =L[U₁]ϕ₁,
          8    2
ϕ₂ =ϕ₀+ϵ (─k₁- ─k₀),
          9    9
            8        17
U₂ =exp[ ϵ( ─𝒵[U₁] - ──𝒵[U₀] )]U₁.
            9        36

第三阶段：

\[
\begin{aligned}
k_2&=L[U_2]\phi_2,\\
\phi_{\rm new}&=\phi_1+\frac{3\epsilon}{4}k_2,\\
U_{\rm new}&=\exp\!\left[
\epsilon\left(
\frac34\mathcal Z[U_2]
-\frac89\mathcal Z[U_1]
+\frac{17}{36}\mathcal Z[U_0]
\right)\right]U_2.
\end{aligned}
\]

这些系数与当前 QUDA 耦合流循环 (/public/home/donghx/PyQuda_a100_2608/quda/lib/interface_quda.cpp:5570) 和 规范场 RK 阶段实现 (/public/home/donghx/PyQuda_a100_2608/quda/include/kernels/gauge_wilson_flow.cuh:90) 一致。

关键是：三次费米子 Laplacian 分别使用 U₀,U₁,U₂，而不是全部使用最终的 \(U_{\rm new}\)。 规范场更新从 staples 构造生成元，经反厄米投影和矩阵指数后左乘原 link，见 link 指数更新 (/public/home/donghx/PyQuda_a100_2608/quda/include/kernels/gauge_wilson_flow.cuh:217)。

因此实际得到的是沿规范场轨迹演化的热流核：

K(t_{f}) = 𝒯exp[ ∫₀^{t_{f}}ds Δ_{B(s)} ],

不能一般地替换成 e^{t_{f}Δ_{B(0)}} 或 e^{t_{f}Δ_{B(t_{f})}}。这个区别在 Note 的 kernel 定义 (/public/home/donghx/Note/Gradient_Flow/main.tex:255) 中也有说明。

## 3. Blending 怎样把 eigenvector 放入流方程？

我们没有显式构造巨大的四维矩阵 K(x,y)，而是计算它对所需初始向量的作用：

\[
\eta_{\sigma n}(x)
=\delta_{x_4,\sigma}V_n(\boldsymbol x;\sigma),
\qquad
\Phi_{\sigma n}(t_f)=K(t_f)\eta_{\sigma n}.
\]

实现流程是：

原始 thin gauge + 时间片 basis
    → 准备四维初始场 η
    → performGFlowQuda：同步演化 gauge 与 η
    → 保存 flowed gauge 和 Φ
    → 构造 generalized vertex
    → 与原来的 perambulator 收缩

由于 Laplacian 不混合 Dirac spin，代码把四个不同 basis 向量放进一个 spinor 的四个 spin 分量中，一次同时演化四个向量。例如 FLOW_MODE_BATCH=32 对应 8 个 spinor。这只是利用 spin 分量作为独立存储通道，并非物理上的 spin 混合。见 向量装填与流调用 (quark_l32x64_expand30_20261012/reference/generate_flowed_vertex_l32x64.py:418)。

Python 层的核心调用是：

performGFlowQuda(
    output.data_ptrs,
    source.data_ptrs,
    dirac.invert_param,
    dirac.smear_param,
    dirac.obs_param,
    nspinors,
)

虽然参数名包含 invert_param，这里它提供的是场布局、精度等信息；这次调用不执行 Dirac inversion。PyQUDA 的绑定直接转发到 QUDA，见 绑定入口 (/public/home/donghx/PyQuda_a100_2608/PyQUDA-gflow-src/pyquda/src/quda.in.pyx:416)。

普通核子 perambulator 保持不变；flow 的两个端点通过 Φ^{†} O[U(t_{f})]Φ 进入 generalized vertex，见 Note 中的 C3 构造 (/public/home/donghx/Note/Gradient_Flow/main.tex:2092)。

## 4. epsilon、nstep 和多流时间如何处理？

当前 L32x96 Blending 入口设置为 Wilson flow，并通过 FlowConfig 指定：

smear_param.smear_type = QUDA_GAUGE_SMEAR_WILSON_FLOW
smear_param.n_steps = config.n_steps
smear_param.epsilon = config.epsilon
smear_param.t0 = 0.0
smear_param.restart = False

见 L32x96 参数设置 (/public/home/donghx/Gradient_Flow/Blending_L32x96/generate_flowed_vertex_l32x96.py:268)。

当前 L32x64 生产脚本明确使用 ϵ=0.01，所以

\[
N_{\rm step}=\frac{\tau}{0.01}.
\]

例如 τ=0.2,0.5,0.8,1.0 分别对应 20、50、80、100 步。其 flow 列表和参数见 当前提交脚本 (quark_l32x64_expand30_20261012/run_expand30_na800_sug.slurm:28)。

框架支持两种调度：

- 单 flow time： 每批初始向量都从 thin gauge 和未流的向量开始，执行完整的 \(N_{\rm step}\) 步。
- 多 checkpoint： 第一段从零开始；后续段设置 restart=True，保留上一个 checkpoint 的 flowed gauge，同时把上一个 flowed spinor 作为下一段输入，只计算新增步数。

例如从 τ=0.5 接续到 0.8，只需新增 30 步，但必须同时接续规范场和费米子场，不能只接续其中之一。见 checkpoint 接续代码 (quark_l32x64_expand30_20261012/reference/generate_flowed_vertex_l32x64.py:693)。

## 5. 时间窗口与边界条件在哪里起作用？

时间窗口不是流方程的定义。代码先在完整四维格子上演化，再只保存源时间片附近的输出：

\[
|t_J-\sigma|_{\rm periodic}\leq w.
\]

同时记录窗口外丢弃的平方范数比例：

\[
f_{\rm discard}
=
1-
\frac{\displaystyle\sum_{t_J\in{\rm window},n,\boldsymbol x,c}
|\Phi_{\sigma n}|^2}
{\displaystyle\sum_{t_J,n,\boldsymbol x,c}
|\Phi_{\sigma n}|^2}.
\]

当前 L32x64 生产脚本要求该比例不超过 10⁻⁶。因此这是对保存与后续收缩的截断控制，不意味着使用三维 flow；也不能仅凭这个范数阈值断言所有观测量的相对误差都小于 10⁻⁶。计算与检查见 时间窗口范数检查 (/public/home/donghx/Gradient_Flow/Blending_L32x96/generate_flowed_vertex_l32x96.py:494)。

费米子的时间反周期边界条件通过时间边界 link 的相位实现；保存用于 Wilson line 的 flowed gauge 时再移除该相位，见 规范场保存处理 (/public/home/donghx/Gradient_Flow/Blending_L32x96/generate_flowed_vertex_l32x96.py:309)。

所以，联系上一问：我们计算的是 K(t_{f})V 的真实扩散结果，不是一组重新正交化的本征向量。flow 的阻尼和范数变化被保留，ringed normalization 也不包含在这段流积分中，而属于后续独立的归一化步骤。
# 固定格距后继的公式与边界

本说明是这批50/20 conf、各200 bootstrap的操作记录。`references/downstream_theory_v2_20261025.md`仅复用其修正版算法/核推导；其中旧20conf/100boot/四动量不描述本次输入。

## 1. 01--06：复用 native 结果，不重新施加重整化因子

对每个 ensemble 独立采用原已算出的链：

\[
M_\Gamma(z,P,t_f)=\frac{h_{\Gamma,\mathrm{bare}}(z,P,t_f)}
 {\operatorname{Re}h_{\Gamma,\mathrm{bare}}(0,P,t_f)},\quad
H_{03}=e^{+d_F|z|/\sqrt{8t_f}}M_\Gamma,
\]

\[
H_{04}=\left[\frac{\zeta_\Gamma(t_f,\mu)}{C_\psi(t_f,\mu)}\right]_{\mathrm{RGR}}H_{03},\qquad
H_{04}=c_0+c_1t_f,\quad H_{05}=c_0.
\]

这里固定候选 \(d_F=0.374137\)，不再乘第二个PT线质量指数；局域点按归一化约束单列为Re1/Im0，不能把局域1硬接入非局域MSbar的短距离对数。
RGR采用原一圈边界+LL，\(\mu=2\,\mathrm{GeV},\alpha_s(2\,\mathrm{GeV})=.30,n_f=3,C_F=4/3\)。

本ensemble的P0、z/a=1,2拟合固定截距b=0：

\[
\ln\frac{H_{05,\Gamma}(z,0)}{C_{0,\Gamma}^{\mathrm{RGR}}(z,\mu)}
 =-m_{0,\Gamma}\frac{z_{\rm fm}}{\hbar c},\qquad
H_{06,\Gamma}^{(b)}(z,P)=e^{+m_{0,\Gamma}^{(b)}z_{\rm fm}/\hbar c}
H_{05,\Gamma}^{(b)}(z,P).
\]

同一原bootstrap行质量作用于两个复分量和全部P；不对moving重拟合。原拟合Q/chi2很差，所有后继继承这一条件性限制。

## 2. 动量、Ioffe time与短距离

\[
P_z=\frac{2\pi n_P\hbar c}{32a},\qquad
\lambda=\frac{z_{\rm fm}P_z}{\hbar c}=\frac{2\pi n_P}{32}\frac za,
\quad \hbar c=.1973269804\,\mathrm{GeV\,fm}.
\]

P0只用于坐标空间与质量拟合，不做LaMET。保留P4,5,6各自原z/a=1..20数据。两ensemble同整数P不代表相同物理P，本轮不插值到共同P/z、不做continuum。

对未测的0<lambda<lambda1，主补全使用非局域固定阶OPE形状：

\[
C_{0,\Gamma}(z,\mu)=1+g[p_\Gamma L_z+c_\Gamma],\quad
g=\frac{\alpha_s C_F}{2\pi},\quad
L_z=\ln\frac{z^2\mu^2e^{2\gamma_E}}4,
\]

\[
(p_V,c_V)=(3/2,5/2),\quad(p_A,c_A)=(3/2,7/2),\quad(p_T,c_T)=(2,2).
\]

偶实部乘 \(C_0(\lambda/P)/C_0(\lambda_1/P)\)，奇虚部另乘 \(\lambda/\lambda_1\)。这是一圈短距离补全处方，不是对未知有限方案问题的证明。

## 3. 大lambda相关拟合与对照

主窗全局固定7<=lambda<=12，27个固定形状：

\[
h(\lambda)=e^{-(\lambda-8)/L}\left[(A+iB)(\lambda/8)^{-a}
+c_+e^{-i\lambda}(-i\lambda/8)^{-b}
+c_-e^{i\lambda}(i\lambda/8)^{-b}\right],
\quad a,b\in\{.5,1,2\},\ L\in\{4,8,16\}.
\]

每形状4个实幅度。Re/Im使用200样本的完整相关矩阵；原rank阈值1e-12，保留模式阈值1e-8，要求rank>4且design满秩，不用对角fallback/prior、不删样本。
中央AIC=chi2+2*4的权重固定传播全部200行。尾部模型spread单独保存，不伪装为bootstrap误差，也不是裸拟合AIC的另一种规则。

主链在lambda9..10平滑拼接；另一个独立分支使用全部已测lambda>=7，在[lambda_last-1,lambda_last]拼接。这是窗口/拼接的联合敏感性检验，不应称单变量变化。图保留所有已测点，包括未进入主尾窗的lambda>12。

## 4. Fourier与一圈matching

采用 \(H(-\lambda)=H(\lambda)^*\)，

\[
\widetilde q(x,P)=\frac1\pi\int_0^\infty d\lambda
[\operatorname{Re}H(\lambda)\cos(x\lambda)-\operatorname{Im}H(\lambda)\sin(x\lambda)].
\]

不对valence再乘2；不丢奇虚部。主积分lambda_max160、分段Gauss12；quasi网格y=-8..8、步长.01。
三通道采用修正v2的非局域MSbar核：V基于1801.03917v2式68，A的额外2(1-xi)为regular项而非plus项，T由2204.08343v4式39 ratio极限与C0_T有限转换得到。逐段公式和独立恒等检查见参考推导及`code/validation.py`。

\[
(K_\Gamma q)(x)=\int d\xi\left[
\frac{f_\Gamma(\xi,|x/\xi|P)+r_\Gamma(\xi)}{|\xi|}q(x/\xi)
-f_\Gamma(\xi,|x|P)q(x)\right]+d_\Gamma(|x|P)q(x),
\]

\[
q_\Gamma^{\mathrm{LC}}=\widetilde q_\Gamma-gK_\Gamma\widetilde q_\Gamma+O(\alpha_s^2).
\]

real-emission使用parton动量|y|P，virtual使用|x|P；不取(I+gK)的精确逆。显式短距离log对应的高|y|渐近实发射补项保留。不再做线质量减除，不引入任意hybrid zs。
输出x=[-1,-.02]与[.02,1]，步长.02；x=0不算，正负两支不跨零连线。

## 5. 输出的物理含义

这是一套局域荷已除去、尚未恢复的条件性shape。不得把connected u/d当完整singlet PDF；u-d需要同位旋简并前提。不能凭本轮的完整数值覆盖把GF_to_MSbar_complete或physics_validated改为true。
源m0与dF/有限方案、强尺度层级、激发态与fallback、尾部、低x/端点和有限P的系统误差仍未控制；没有做连续极限或P->infinity外推。

生成日期：2026-11-01；标题：两个ensemble独立固定格距的quark光锥诊断。

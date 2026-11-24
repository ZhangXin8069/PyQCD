# 09 light-cone：LHAPDF 与 JAM 比较入口

两ensemble独立固定格距；原09数值、200样本和归一化未修改。

## L32x64

- [unpolarized u-d：三动量叠图，q与xq](L32x64/09_lightcone/lhapdf_comparison_20261101/plots/overlay_vs_LHAPDF_unpolarized_u_minus_d.pdf)
- [helicity u-d：三动量叠图，q与xq](L32x64/09_lightcone/lhapdf_comparison_20261101/plots/overlay_vs_LHAPDF_helicity_u_minus_d.pdf)
- [transversity u-d：JAM条件性形状对照（非LHAPDF）](L32x64/09_lightcone/lhapdf_comparison_20261101/plots/transversity_u_minus_d_vs_JAM2022_shape_only.pdf)

## L32x96

- [unpolarized u-d：三动量叠图，q与xq](L32x96/09_lightcone/lhapdf_comparison_20261101/plots/overlay_vs_LHAPDF_unpolarized_u_minus_d.pdf)
- [helicity u-d：三动量叠图，q与xq](L32x96/09_lightcone/lhapdf_comparison_20261101/plots/overlay_vs_LHAPDF_helicity_u_minus_d.pdf)
- [transversity u-d：JAM条件性形状对照（非LHAPDF）](L32x96/09_lightcone/lhapdf_comparison_20261101/plots/transversity_u_minus_d_vs_JAM2022_shape_only.pdf)

CT18NLO：gV=(2,1,1)；NNPDFpol2.0：逐成员截断轴荷，xmin1e-6；JAM：固定截断正x中心积分。
参考带分别采用native Hessian 68.27%CL、replica SD、JAM点误差；不加到格点统计带中。
q图y限[-0.5,4]；正x仅quark，不减antiquark。格点是局域荷归一化shape，不是绝对PDF。
所有原dF/m0/有限方案和重构限制保留，未做连续极限。

[代码、公式、运行说明](lhapdf_comparison_20261101/README.md)
[独立验收](lhapdf_comparison_20261101/verification.json)

日期：2026-11-01；标题：两ensemble的09 Lightcone参考比较。

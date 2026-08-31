# Explicit straight-line OPE channel forward eval

## 场景

独立 `gpt-5.6-luna`（reasoning `max`）评估器只读
`pyqcd-tmd-algorithm/SKILL.md`、`references/geometry.md` 与
`pyqcd-conventions/SKILL.md`，不得读取实现或测试。真实请求是：为现有直线胶子 OPE
新增 unpolarized 通道，同时保持 docker-v20260805 完全兼容，并支持 `±z`、复数输出、
多 Lorentz 对和低显存执行。评估器须给出可执行 API、物理边界、缓存所有权与验收门，
并逐项判定文档是否足够无歧义。

## RED

首轮能正确区分 legacy `F·Ftilde`、unpolarized `F·F` 与 helicity `F·Ftilde`，也拒绝
凭空假设 `O(-z)=O(+z)*`；但发现文档仍缺：

1. `OPEChannelSpec`/wrapper 的完整签名、返回 shape/dtype 与异常；
2. `bare_spatial_sum` 的确切除因子边界；
3. `zdir=2` Lorentz 指派和新通道组合系数的职责；
4. 所有直线 mode 的去迹选择、signed-z 编码和 cache 借用所有权；
5. 可执行 contract、oracle 容差与完整 docker 磁盘验收规则。

## GREEN

补全后，同一评估器 fresh reread 对前四组及小格点 contract 全部给出 PASS：公开调用为
`FieldStrengthCache + OPEChannelSpec + gluon_ope_channel`；输出固定
`(delta_z,Nt)` complex dtype；裸归一化为逐时间片全空间色迹求和且不除体积/`Nc`；
`zdir=2` 的三个单通道为 `(3,0,3,0)`、`(3,1,3,1)`、`(0,1,0,1)`；当前所有直线
mode 只接受 `legacy_untraced`；`delta_z` 只计 `|z|`，符号由 `direction=±1` 给出；
cache 返回借用对象，`max_entries` 只约束 cache-owned 引用。

第二次复验只剩完整 docker 门不够机械。再补入固定命令、函数级 `<1e-10`、磁盘级
norm 相对差 `<1e-6`、shape/NaN/零分母规则、十组态文件清单、退出 `0/1/2` 语义及
“无独立 hash、不得编造”的边界后，最终逐项结果为 7/7 PASS、无残余缺口。

## 边界

这是技能文档的前向行为证据，不是生产实现或真实数据证据。实现分别由
`python -B -m pyqcd.testing._ope_channel_contract` 与
`python -B -m pyqcd.testing._field_strength_cache_contract` 验收；完整 docker 产物缺失时
一致性脚本应退出 `2`，不能由该 eval 升级为数值兼容 PASS。

# lqcddb 来源与 PyQCD 实现状态

## 1. 状态总表

| 对象 | lqcddb source | PyQCD 当前包 | 未闭合边界 |
|---|---|---|---|
| syntax/API audit | implemented, documented | source-specific, 不作为运行依赖 | README/type stub/source 可能不一致 |
| eigenvector Gram/compression | implemented, partially tested | 有 compression 与 dtype contracts | 历史数据 provenance、随机高模补全 |
| momentum phase | implemented | oracle tested | basis phase vs vertex `q_sink` 必须分开 |
| VdV/VVV | implemented | readers/vertices tested | link/displacement/conjugation 实例 |
| perambulator reader | implemented | binary reader tested | fixed legacy format 和 byte order |
| sequential perambulator | implemented | formula/port present | 每个 producer 的 source/sink orientation |
| Wick expansion | implemented, hand-oracle partial | PJN 3pt four-diagram oracle passes | PJNNJNP 4pt no independent 12-diagram oracle |
| dynamic equivalence | source heuristic | default opt-in/closed | gamma/time/topology equivalence not proved |
| GEVP | source real-projects complex matrices | complex Hermitian fixed and tested | state tracking、conditioning、真实矩阵 |
| Stout smearing | source has known implementation risks | port tested identity/unitarity/action/backend | historical production equivalence not established |
| contraction adviser | heuristic | diagnostic only | ellipsis accounting/efficiency not hardware-validated |
| MPI domain transport | source module imported eagerly | separate `_mpi_transport` port | not the main reliability path |
| MPI meta-task pipeline | n/a | implemented, tested | TMD parallel step intentionally rejected |
| HDF5 I/O | source legacy helpers | canonical PyQCD write/load | external legacy files require provenance |
| legacy distillation pipeline | reference outputs | closed at 237/237 regression scope | physical validation per observable separate |

## 2. 已关闭的回归范围

PyQCD 蒸馏回归以冻结 legacy baseline 复现 10 组态 9 步：

```text
vertex -> 2pt -> ope -> 3pt -> 4pt -> analysis -> plots -> report
```

入口：

```bash
python -m pyqcd.testing.regression.distillation.main run --conf-ids 6250
bash pyqcd/testing/regression/distillation/run-local.sh
```

全量记录为 237/237 PASS，中间数据和分析结果按注册容差比较。这个 `closed` 仅表示
legacy workflow regression 和 artifact contract 闭合，不表示 PJNNJNP 4pt、Stout、
GEVP 或真实物理量已经 independently physically validated。

## 3. 已知 source limitations

以下限制来自只读 `skills/donghx/lqcddb` 和 `skills/sush/lqcddb` 来源审计，不应直接
复制成 PyQCD 生产行为：

1. `identify_equivalent_diagrams()` 的 normalization 主要比较 perambulator index
   structure，不证明 gamma tensors、vertices、time labels 和 connected topology。
2. `ignore_dis` 是 heuristic；disconnected topology 必须从 contraction graph 判断。
3. current GEVP 曾 symmetrize 后取 real part，破坏 complex Hermitian matrices。
4. current Stout 的 documented rank、internal transpose、zero-generator branch 和
   spatial roll-axis mapping 有风险。
5. `ratio_3pt` 的 axes 可占据不同位置，当前广播可能产生错误 shape。
6. dynamic module 在 module load 时 import MPI helpers，wildcard import 可能初始化 MPI。
7. `TGather` 要求 equal local leading sizes；size-one 和 multi-rank helper shape 不一定
   相同。
8. `write_data_ascii` basename-only path 会尝试创建空目录。
9. bundled `savetxt` 依赖 legacy `numpy.compat`，需要当前 NumPy round-trip 检查。
10. `cached_contract` 的 ellipsis shape validator 与 NumPy 在部分合法 broadcast 情形
    不一致。
11. performance adviser 的 presets 可含 `efficiency>1`，ellipsis accounting 可能漏
    batch dimensions；bandwidth/time 只是 heuristic。

## 4. PyQCD 已补强的合同

- `solve_gevp` 保留 complex Hermitian eigensystem；`test_gevp_preserves_complex_hermitian_data`
  与 `scipy.linalg.eigh` 对照。
- momentum-smearing phase、VdV/VVV 和 rectangular lattice 有独立 loop oracle。
- `readin_eigvecs`、`readin_peram`、VdV/VVV reader 有 round-trip/shape/byte-order tests。
- HDF5 `save_tensor_h5/load_tensor_h5` 是 canonical，保存 backend/complex dtype 语义。
- PyQCD Stout 有 identity、`rho=0`、unitarity、action decrease、time-chain 和 backend
  contracts。
- PJN 3pt 有四张手写 Wick 图的 independent oracle。
- MPI meta-task 有 planning、run-dir、GPU binding、failure propagation、cache/recompute
  和 rank0 输出 contracts。

这些补强只覆盖各自合同；不能把某个 unit test 的存在传播到其他拓扑。

## 5. 仍未实现或未验证

- PJNNJNP 4pt 的 independent 12-diagram oracle。
- dynamic equivalence 在完整 gamma/vertex/time/topology 证明下的启用。
- Stout 与历史生产 smearing 的逐位/物理等价。
- 通用 complex GEVP state tracking、`t0` 稳定性和真实系综 validation。
- domain-decomposition transport 与主 `pyqcd.parallel` 调度语义的统一。
- 对所有历史 eigenvector/perambulator 文件的 producer provenance。
- 蒸馏链的通用 `physical-validated` 标签；必须按具体 observable 单独建立。

## 6. 验证命令

仓库级：

```bash
python -m pyqcd.testing
python -m pyqcd.testing.regression.distillation.main run --conf-ids 6250
bash pyqcd/testing/regression/distillation/run-local.sh
python -m pyqcd.testing.regression.gevp.verify
```

静态/语法：

```bash
python -m py_compile pyqcd/contraction/_autowick.py \
  pyqcd/contraction/_dynamic.py pyqcd/contraction/_seqperam.py
python -m py_compile pyqcd/tools/_io.py pyqcd/analysis/_analyse.py
```

如果真实 PyQUDA 环境不可用，只能报告 serial/static contracts；不能声称真实求逆、
perambulator 或 MPI production 已完成。

## 7. 交付状态模板

```text
object:
physical definition:
conventions:
tensor formula and axes:
algorithm/plan:
implemented evidence:
tested evidence and tolerance:
closed scope:
physical validation status:
known limitations:
```

最高状态只取最后一项证据支持值；缺失项必须留空或写 `not established`。

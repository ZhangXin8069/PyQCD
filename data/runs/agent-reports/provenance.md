# PyQCD 外来内容来源、范围与风险审计

审计对象为 `dev20..HEAD` 及 `dev20` 快照中的拼音参考树，重点 `refer/donghx/_Gradient_Flow_GluonPDF`。审计时 HEAD 为 `3e1edd4f9d9eadd3ea07dfc47e4e92cd2fc6c0b9`；初始工作树仅见未跟踪 `data/runs/`，本任务未修改 `refer/`、生产代码或 Git 索引。

## Git 范围
- `dev20=3d528be28ca60d2581b50a43ae3704afb5061e85`。严格 `dev20..HEAD` 共 5 个提交、6151 个路径变化、583 行新增、1,996,386 行删除；其中 `refer/git-rep` 6134 个路径主要是普通拷贝到 gitlink 的转换，`pyqcd/` 7 个路径是后续本地修复。
- 四个现存拼音树在 `dev20..HEAD` 的路径差为 0：`git diff --name-only dev20..HEAD -- refer/donghx refer/huangcl refer/sush refer/zhangxin` 无输出。故严格区间没有新增可吸收的拼音源码。
- `dev20` 提交自身只改 25 个 TMD/测试/日志文件，未改 `refer/`。所谓“`dev20` 快照内容”是快照内既有内容，不是 `dev20` 标签提交引入。
- 重点目录实际由 `fc3a7c915a267f625512c220f805435df3728dab`（父 `08e8f418...`，作者 `zhangxin`，2026-10-20）一次性加入：`181 files changed, 27510 insertions(+)`，其中 16 个 PNG 为二进制；`dev20` 与 HEAD 均为 181 文件，且 `git diff --quiet fc3a7c9..HEAD -- 重点目录` 返回 0。
- `refer/AGENTS.md:11` 声明 `refer/zengch/`，但 HEAD 中该路径不存在（`test -e` 返回 1）。本报告不虚构该来源。

## 重点目录盘点
- 181 个跟踪 blob、4,460,308 字节、14 个 `100755`、0 个符号链接、0 个单个大于 1 MB 文件；聚合清单 SHA-256 为 `47bd7308e2c8c37ea168a3c634eb5c812541fe12d2a6b4ee17807d3aa730456a`。
- 类型：111 Python 脚本、40 普通文本、16 PNG、10 Shell、2 TeX/文本、2 C/C++ 源。无 ELF/归档/`.npz/.npy/.csv` 数值产品；PNG 是生成图，不是可复算数据。
- 入口证据：`README.md:18-22` 指向不存在的 `analysis/helicity_operator_comparison.py`；`full_chain.py:13-15,58-100` 依赖缺失的 `reference/`；`pipeline.py:17-23` 硬编码已不存在的公开项目根；`ope_components.py:19-20,51-60` 给出 OPE/规范场合同；`run_pipeline.py:67-97` 启动 4 GPU/MPI。
- 来源证据：内容硬编码 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF` 与 `/public/home/donghx/...`（如 `run_quark_bare_20.py:56-58`、`run_gpu.sh:4-15`、`ope_components.py:58`），指向当前已缺失的 Dong HX 集群树；没有 canonical URL、上游 commit/tag、CITATION/AUTHORS、依赖锁或构建清单。
- 许可证据：目录内没有 `LICENSE/COPYING/NOTICE`，也没有 `pyproject.toml/setup.py/requirements`；`git ls-tree ... | rg 'LICENSE|...'` 返回 1。Git 导入提交作者仅是导入者，不能替代版权许可。
- 工程形式是扁平化、裁剪后的工作区快照：没有子目录，文档引用的 `reference/`、`analysis/`、flow cache、NPZ/CSV 均未随 181 文件保存；多份文档与脚本因此不可独立复现。

## 来源与安全风险
- 供应链结论：`origin=partially attributed / upstream ref=unknown / license=unknown / snapshot SHA-256=fixed`；当前只能只读参考，不能复制进生产路径。
- 25 个文件含 `subprocess/os.system/eval` 类进程或动态执行标记，13 个文件含 `rmtree/rm -rf/unlink` 类删除标记；合并扫描有 39 个执行/清理文件。重点风险包括 `full_chain.py:31-39,101-105` 可删除 vertex/cache，`pipeline.py` 复制后清理临时树，`run_pipeline.py:113-125` 删除 `/tmp` scratch，多个 `submit*.py/.sh` 调用 `sbatch`。
- `build.sh:5-10` 会加载模块、编译未审计 C++ 并执行 self-test；`run_gpu.sh:4-15` 会 source 外部个人环境脚本并切换目录。两类命令均未执行。
- 未命中 `shell=True`、`curl/wget`、网络请求库、`pickle`、`torch.load` 或常见凭据/私钥模式；凭据模式扫描四个拼音树的匹配文件数均为 0。部分 `np.load` 未显式写 `allow_pickle=False`，任何外部 NPZ 仍应按不可信输入处理。
- 风险定级：许可/来源为 Important 阻塞；删除、作业提交、外部环境 source 和缺失依赖链为 Important；未发现 Critical 秘密或已知恶意载荷，但二进制 PNG/C++ 未做内容取证。

## 候选处置矩阵

“keep”仅表示保留为有明确许可的参考树，不表示可由生产代码 import。

| ID | 文件数 | 处置 | 本地缺口与重复能力 | 来源可信度 | 验证方式 |
|---|---:|---|---|---|---|
| D-GF | 181 | reference-only | 高价值：流化 OPE、Clover 归一化、HELICITY `TH-SH`、共享 bootstrap、Fourier/matching 诊断；本地 `_gradient_flow_gluon_ope.py`、`_helicity.py`、`_tmd.py`、`_tmdextract.py` 已覆盖核心 | 中低；来源指向缺失集群树，无许可 | 弱场 Clover、`z=0`、±z 宇称、共享样本、Fourier 归一化、路径删除守卫 |
| D-META | 13 | reference-only | `Note/`、`docs/`、根说明；2pt/OPE 已在 `pyqcd.operator/pipeline` 与 R8 覆盖 | 中；作者树但无项目许可 | 只核对公式/路径；不复制 |
| D-CODEX | 134 | reject | 归档 agent/skills/运行配置，与 PDF 生产无直接缺口，且配置陈旧 | 低；工具快照、非算法来源 | 仅保留目录，不载入、不执行 |
| H-CORE | 139 | reference-only | ratio、拟合、FH、GEVP、2pt/3pt 分析；已映射到 `_ratio2pt/_ana_ratio/_bare_matrix/_proton_energy/_fh/_ana_3dir` 与 B10 | 中；`refer/AGENTS.md:9,24` 有归属但无 LICENSE/URL | 合成 `meff/E0/c0` 回归，API 对照 |
| H-META | 14 | reference-only | docs/本地分析产物及 1 个 Codex 符号链接；无新增算法 | 中低；历史分析产物 | 文档路径与引用完整性检查 |
| S-CORE | 240 | reference-only | Wick/蒸馏工具；R1/R2/R3/R6/B7/B8/B9 已吸收，剩余大量 Slurm/重复生产壳 | 中；`refer/AGENTS.md:10,23`，无项目 LICENSE | 原版真值已由现有契约覆盖，新增项需隔离测试 |
| S-CODEX | 65 | reject | 归档 agent skills，与蒸馏核心无关 | 低；工具快照 | 不加载、不执行 |
| S-META | 7 | reference-only | docs、根说明；作为历史解释材料 | 中低 | 文档完整性检查 |
| Z-CORE | 23 | keep | MIT；完整胶子 PDF 工作流与分析框架。B2/B3/B4 已吸收，保留为许可清楚的参考 | 高；`LICENSE:1-17` | 对照 `quasi_pdf_gluon`、OPE Lorentz 表、parity/boundary |
| Z-BUILD | 86 | reject | `.report-build.hgOTls/` 页面 PNG/TeX stdout，可重建且非源码 | 低；生成物 | 不吸收，不复用缓存 |
| Z-DOCS | 3 | keep | PDF/TeX 报告，作为来源证据 | 高；MIT 树内 | 哈希与引用完整性 |

候选统计：905/905 个现存拼音跟踪文件已分组；`keep=26`、`reference-only=594`、`reject=285`、`adapt=0`。当前无 `adapt` 是因为重点目录许可门未通过；取得明确许可和上游固定 ref 后，D-GF 的独立测试/公式可实现部分可重分类为 `adapt`。

## git-rep 边界

| 路径 | gitlink OID | README 声明的来源 | 当前许可证据 |
|---|---|---|---|
| `EasyDistillation` | `f74bf7de5ebb3ae94107c147339f66c5552cf240` | https://github.com/IHEP-LQCD/EasyDistillation | `LICENSE:1-3` MIT |
| `LQCD_Master` | `e4a443e08d5904371fc280ce93c00ff84df21893` | https://github.com/sjtu-sai-agents/LQCD_Master | 顶层未找到 LICENSE，未解决 |
| `LQCD_Skills` | `90bcbb8739c80b41a3965f41dab50bdfc9841b16` | https://github.com/AI4LGT/LQCD_Skills | 顶层未找到 LICENSE，未解决 |
| `PyQUDA` | `138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` | https://github.com/CLQCD/PyQUDA | `LICENSE:1-3` MIT |
| `lamet-agent` | `9355c52ab7f5061fcc1befacfb7e1b65a0b32db1` | https://github.com/Greyyy-HJC/lamet-agent | 无 LICENSE 文件，`pyproject.toml` 未给许可，未解决 |

`PyQUDA` 嵌套 `pycparser` gitlink 为 `5b60167602e7914139b3e277266ccdff5c24dbf2`。来源表见 `refer/git-rep/README.md:7-17`，URL 也见 `.gitmodules:1-15`；`420790a` 将四个普通拷贝转成 submodule，`05f9080` 将 LQCD_Skills 从 `skills/` 移到 `refer/git-rep/`。本次不进入子模块做功能 debug，也不执行其测试。

## /public 数据核查

- 配置：仅 `beta6.20_mu-0.2770_ms-0.2400_L24x72`，11 个 LIME，`4150,6250,...,8050`；每个约 573 MB。
- 本征系统：同一系综 11 个组态，每组 72 个 `eigvecs_tNNN_<conf>`，共 792 文件；每个约 66,355,200 字节。
- perambulator：同一系综 11 个组态，每组 288 个 `perams.<conf>.<d_source>.<t_source>`（4 源方向 x 72 时间），共 3168 文件；每个 46,080,000 字节。
- 文件级门：默认 10 组态 `6250..8050` 均为 `gauge=1/1, eig=72/72, peram=288/288`。`pyqcd/pipeline/_config.py:25-38,126-139` 的名称合同与这些路径一致，因此静态上具备重跑 `python -m pyqcd.testing.tmd.pdf_workflow` 的 gauge/eigen/peram 输入。
- OPE：现有 `/public/group/lqcd/donghx/Ope_Gluon/TMD_Ope_Gluon/Result/...` 只有 conf `4150`，x/y/z 分别 6/6/46 文件（共 58，含 1 log、57 NPY）。默认 10 组态没有预计算 OPE，但可由 gauge 重新计算；不存在 `Result_hpy_4D_10times` 旧声明路径。
- 与声明路径差异：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF`、`/public/home/donghx`、L32x64/L32x96 系综均不存在；因此重点目录当前无法按其脚本重跑。`/public/home/sush/distillation/{lqcddb,function_contraction}` 是指向 `/root/lattice-pdf/examples/sush/...` 的外部/不可验证符号链接，本审计未跟随。
- 机械结论：对 pyqcd 的单系综 L24x72 真实链，输入数量足够做静态预检和冒烟；对重点 Dong HX L32x64/L32x96 链，输入不足且快照依赖缺失。
- 物理解释：只有一个系综、一个格距和 10/11 个组态，且无跨格距数据，不能据此完成受控连续极限或“最终物理 PDF”。它是真实数据链回归，不是完整物理闭环；无 checksum manifest，二进制完整性仍为“未验证”。

## 证据命令与退出码

```text
git rev-parse HEAD dev20; git log --oneline dev20..HEAD                          rc=0
git diff --name-only dev20..HEAD -- refer/{donghx,huangcl,sush,zhangxin} | wc -l rc=0, 输出 0
git diff --quiet fc3a7c9..HEAD -- refer/donghx/_Gradient_Flow_GluonPDF          rc=0
git diff fc3a7c9^ fc3a7c9 -- 重点目录                                             rc=0, A=181
git ls-tree -r -l HEAD -- 重点目录; 逐 blob SHA-256                               rc=0, 181/181
git submodule status --recursive                                                   rc=0
rg 秘密/私钥/常见凭据模式 四个拼音树                                                rc=1, 匹配 0
rg shell=True/网络/pickle/torch.load 重点目录                                      rc=1, 匹配 0
find -P /public/... 配置/本征矢/peram/OPE；逐组态计数                               rc=0
git status --short --branch                                                        rc=0, 初始仅 `?? data/runs/`
```

审计期间工作树出现其他并发写入（`pyqcd/`、`docs/plans/` 及其他报告）；它们不在本任务创建范围，未读取其 diff、未修改亦未回退。
未运行任何 `refer/` Python、Shell、Slurm、C++ 或 pyqcd 数值工作流；原因包括只读边界、外来许可未知和禁止执行要求。

## 禁止执行

- 禁止运行 `build*.sh`、`flowed_gluon_ope.cpp` 编译/self-test、任何 `run_*.py`、`submit*.py/.sh`、`full_chain.py`、`pipeline.py`、`run_pipeline.py`。
- 禁止执行任何含 `sbatch/mpirun/rmtree/unlink` 的外来入口，也禁止 source `/public/home/donghx/*` 或跟随 `/public/home/sush/distillation/*` 外部链接。
- 禁止 `pip install`、网络拉取、递归初始化/更新 submodule、修改 `refer/` 或对 git-rep 子模块做功能 debug。

## 拟议本地适配测试

- 在 `pyqcd/testing/_gradient_flow_gluon_contract.py` 增补弱场 Clover 归一化：`F00=sin(flux/2)`、`2Tr(F^2)/flux^2 -> 1`。
- 在 `pyqcd/testing/_tmd_su3_geometry_contract.py` 增补 ±z 宇称、unpolarized `TU-SU`、helicity `TH-SH` odd 与 `TH+SH` diagnostic 分离。
- 在 `pyqcd/testing/_gradient_flow_gluon_contract.py` 增补 `tau=0` 恒等、RK3 单步确定性、flowed link 幺正性；不得直接调用外来脚本。
- 在 `_quasi_tmd_fourier_contract.py` 增补 cos/sin 变换、`x->0` 保护、`H(0)=2<x>_g` 归一化和共轭对称性。
- 增补共享 bootstrap 索引、逐样本真空减除、零信号合成数据 `c0=0`、坏样本 NaN 隔离测试。
- 增补路径安全测试：拒绝符号链接缓存、缺失输入、外部 root 写入和清理；失败时必须保留输入。
- 适配先决条件：补齐上游 URL/固定 commit、作者授权/许可证、原始树路径和 data-product hashes。通过后先跑 `python -m pyqcd.testing`，再跑单组态 smoke。

## 未解决来源与许可

- D-GF 的精确上游 commit/tag 和许可证未知；`fc3a7c9` 只是本仓库导入点。
- Dong HX、Huang CL、Su SH 三棵拼音树均无项目级 LICENSE/URL；当前只能 `reference-only`。
- LQCD_Master、LQCD_Skills、lamet-agent 在固定 OID 上未找到许可文件，不能升级为 `adapt`。
- `/public` 的 LIME/eigen/peram/OPE 大文件无随附 manifest 或逐文件 SHA-256；只确认了路径、数量、大小和命名合同。
- 重点目录的 16 PNG、57 个外部 OPE NPY（仅 conf4150）和所有外部快照均未做数值内容验证。
- `refer/zengch` 文档声明与现存树冲突；来源是否被误删/改名为未验证。

## 重点目录 181 文件 SHA-256 清单

格式：`sha256 size mode relative_path`；覆盖 HEAD 的全部 181 个跟踪文件。

```text
c97e2836b76e22a52066de9d2662b489600297ee2558284ca00ee2183a0976d9 3598 100644 INTERPRETATION.md
07214ef117a0606fd9e15eed57a6b43d25fe7172d9af15cfcf4345af316355fa 28078 100644 L32x64_divergence_Zchi_test_20261015.md
e97e1a07955f6189f614483aca3f83640fe731e3ffa40fe278a1d544128329f9 12717 100644 NO_Z0_NORMALIZATION.md
fa5db361cdeca8b3872b1e81d80a153d054207f18681505816fb73e5600806f6 1077 100644 README.md
ecafae563398513f41744c9b40057af630c05e8d4fe6c4b4b14ecfb995929474 2645 100644 RESULTS.md
b8adc6e37f020dde67b53990dd1b9ae9f8aeb9e05571c7273a91ac5bcdee7dd2 1478 100644 UNIFIED20.md
bc0056400ffaad08fe91c5d91eff7f1d3ea3b25db528f0b3207aff30c786099f 2069 100644 VALIDATION.md
29580eb688e38aefed4cc0aaa64d78590e1a89a2b239ed0950af12d9b9778738 75 100644 __init__.py
d3b9436efb3057b078aa2c3c1e090a14d6247eda1e0b2fc5b5a6b55f6825722f 209726 100644 aic_extrapolated_m0_vs_zpz.png
32070f8fe4b1046635ba3084e03185949aea820b1733131f16c8a93f2ab595c1 16022 100644 allconf.py
9866a7614bac46598b6dba03046adacbbe4d2962eaff4dac772286ee349f54ed 52004 100644 analyze.py
8644657ad33a3a2c138c3fbeb2c0483f033f2c234f4f7a5636b66c4a5294e8b4 18414 100644 analyze_tau_gt_0p1.py
a7a8be4e430c292f3522b61f949c14519a9481a2eae15f31bf483035f86f8d5f 3320 100644 audit_cpu.py
759fa644ee53bf9531462b76208f31efedd03e090da52fdd0fa4da9cc7355fdd 3251 100644 audit_inputs.py
d711c04fe4416c11a3ed7b29d562db82b71e2ba7e5134e7b6db6bb9906c7b8de 3098 100644 audit_missing.py
f8429316030f851db4b91b7739bb9a9a864e0af417fb42436222abd06793e750 1674 100644 audit_paths.py
fed7ba6df2acc38180daa08cc6d726c1e9c95de16313a755c47d36784d6517a0 20853 100644 audit_progress.py
32021b7ed2bcc9c827c5a2fc1fc7851d2d6a7f9a4e9233ed0a1341b35a0f9d4f 1214 100644 audit_results.py
f20d73a681cade5eb08c93b06045973f8671d901fd2240f86f73e5322bfed21a 3212 100644 audit_statistics.py
7b1277ca6572da5622d91a2b019ba20306b1a4076f7aacbfc4e12753ab17a8b7 2282 100755 audit_tf03_tf04.py
e7bcf9f144c753b9502d2b7807664d2de0c8afa52eb0a5eb4efe622fb15e88ce 136799 100644 bare_matrix_vs_flow_z2_all_flow.png
9df0fbdf2b199f32e67d350e663cdd8780868a35ac3d9090c80924469680f7a7 140489 100644 bare_matrix_vs_flow_z2_all_flow_fullrange.png
e377a18346b4ae55c9eea21c4a82b4462cd50e45cb89a4e2c7ec2636e57138cf 31296 100644 basis_vertex_20261014.md
e377a18346b4ae55c9eea21c4a82b4462cd50e45cb89a4e2c7ec2636e57138cf 31296 100644 basis_vertex_before_weights_20261015.md
f8702e70ce7ea4bdcea31e6849b53ab7d02a5da3e0cd6e3562d47dd6406a2570 53873 100644 basis_vertex_weights_20261015.md
bc2618deeaf80b0e7b7e27ecb4a1859c99fac84e0ed0869017a42fffdc31e1a9 10298 100644 benchmark_conf10000.py
a3fbc84c2832d0b5b82b6817e71395178383d501cb62a64a9d0ab8955227cbc3 388 100755 build.sh
9cd14b38e379b379d84bface3ec15a955b1e611d5390f630cf2ca93415e2fe5a 5137 100644 build_unified_20.py
9215dbc9de265bed4286294d257abdfefd97a7b0b560e2d8a7fe628ef70bbdfe 534 100755 build_v3.sh
71f342d50055fb7bae55767128870e48d5d591d9dea263090a26a8fcf48e6075 537 100755 build_v5.sh
efc7807557129b6ed566a6e7970c3f73776c3e4707a89e434eef03ce0d20e8bd 136075 100644 c3c2_midpoint_vs_flow_z2_tsep6.png
4da2718dc3c95b692a61750754f49a8dce2a089fac0bf3623854d8f7db909585 149681 100644 c3c2_vs_insertion_helicity_TH_odd_imag_z2_tsep6_all_flow.png
f1a6e2185efac85271e9d739bfd674408495ea3a3e074cd8df89f6191a098d73 150482 100644 c3c2_vs_insertion_helicity_TH_plus_SH_odd_imag_z2_tsep6_all_flow.png
e5a16969447fe570eb520d90e335766c1c3f20cd86fdcff9adf09630f61c2eed 152901 100644 c3c2_vs_insertion_unpolarized_TH_minus_SH_even_real_z2_tsep6_all_flow.png
1ff5e2c6cacea9b595b9dbf6b79e7637dd08ca70a20e95922d84b1a48d5adc7b 25181 100755 calc_ratio_matched.py
7c3abe24d8c19f56e7e106b3c7c31574448cfe086b0c0b535bf5c6b399228724 7728 100644 campaign.py
2d20f5a8e61c1bcb3ae47fab8a8dd4233a393854064f70c1fac67a44157cc600 999 100644 check_clover_normalization.cpp
5dc4ef06f9027e504d41e724ce8566208f0ec0c84cbee4a0267f175dd9bf5223 3836 100644 check_solver.py
414ebc18852c46e4f82377a73cd1f02f89470e178e37ffa3804005c9528e23ce 7992 100644 cleanup_completed_cache.py
d13ed1223846e4fa539b5dbdedf2d20f4a8efa135d4dfdd4f53cd39a78b044f3 3299 100644 cleanup_l32x96_vertex_only.py
659908d75e3262cf4bec32a6b6d53eb5cfce82c1ea46c18b1f46c210d55d4bec 8641 100644 cleanup_recoverable_cache.py
7b7b5c6d63b6d2d1aeef7c9a51b5add3738d08d0baf032731fac857c13bf061d 9179 100644 collect.py
001745946a415b1ff8f93006ee9647b46096f843628528b23e6e54b1df1c3977 4182 100644 common.py
f42bbb69fba4626c008ea9af63f2b7ddb172c5de8d244946dac99cdad8d1628f 13843 100644 compare.py
09561d91d41dc758d9b8cc14019e976b9e6bc998319357f19c96821f6fcdd817 2382 100644 compare_epsilon.py
6b90eae02b4c087986258de8927d13c32d1574cda98f0016c4f028de50fa5717 16669 100644 compare_windows.py
7e11ffb839b7a3a404635016e7456dda9f99bd17c3d63a1dde13b4c53f989156 19615 100644 construct_bootstrap.py
95d57bd009696ed39d30380ea600ba6549a99d2c6ec351a689fa0378d28a26c1 7551 100644 core.py
b42ea21e880b25c6a7a423f18e8f35b3165d6ae8987bba6a99c6a0421235a12f 1693 100644 cpu_gate.py
3403335472efa6526b8e887a34623318e0078327e904a2e056a90bc263283bbb 35374 100644 divergence_20261014.md
b0b88bf16c60483457b6ce5a5f677ebdd593cfa848df939b7de89c37771a2ca1 17919 100644 extrapolate_matched_small_flow_tau_ge1_v1.py
d0e33c06dcbfcabb54956fbe7c6d2488f125f47c7ec246baf9af789f31b3d4b6 20497 100644 extrapolate_unpolarized_tau_ge1_correlated_v2.py
71da63a791b4fb87c51c7c52ee23c8c4a83838247fbd41fb7ab4faabc5e63a5a 11138 100644 fast_snapshot.py
4da59cc0de9bc12eeea5e280c65ee64571cb1a2b14a15be1f16a979cbda6b462 7927 100644 finalize_progress.py
776a9f9cc1fad10df283baa585368530de3d9d523923d47f9e90ac3b24a7c082 4508 100644 finish_recovery.py
c5ee96b56c760581810266e74a914bceb3e4f18edc553b002320377e19434b7d 35294 100644 fit_all_operators_aic.py
029dc99810e8f064d6e567a223887a2f4a33760232578259a3ecc0ae053c80ab 16581 100644 fit_bare_latest.py
7482f377edfbdc7f14dfae8094bde892ba661558cea225d042b2e5d91e2bbfe9 13964 100644 fit_cpu.py
3706523d7a1d9b983544ee6b66cabbbbdc1c97a4065c84586afd6cdbb342c72d 35411 100644 flow_equation_20261013.md
e84636928ba526c342e9ff2315cc00dde6ae3bf4baefb644350d9635a5a6ce7a 7986 100644 flow_equation_before_omega_20261013.md
fee4830ad409ef5e622b499945d09ab0845bba73e776d633c001f8e3946b3a26 27108 100644 flowed_gluon_ope.cpp
7e242f7dafefd2f2bbe9878cac36398a3d55058cb97e6f00e670f2d965ee79f3 3331 100644 freeze_memory.py
77fe0ff80384032cd3d775efaa3dd29e8dfb7903b3932a9b64d98d58f31009ec 1399 100644 freeze_request.py
a5e20d0dbf0b00c549d718424ebbf515d290bb08ac6f50f46a988b4d4b8598fd 6788 100644 full_chain.py
a5ddcd798eea4f05fb04af0f1d24c3f7c44e12cf3e97b921b3aa0123df082f7b 7727 100644 generate_selected_basis.py
5aba8cfa5151d8b81bb8ff6ce9271f36722219cb968da5a27cde09a714e2d5d6 6193 100644 gluon_absolute_normalization_v1.py
f09d80ada7df4e1a6077b5e9a867c6c2b3781e0d9098dd9ecbe1e0d290d120e0 6851 100644 gluon_flow_to_quasi.py
e620b7a29f27a1c3bef1e8b4a386c165acf19a367fb030d1b80ed4ca76bc2ef2 12159 100644 gpu_worker.py
1a6d402ada43e2e66a3f7ea53ea205b607e368cc93466f0cad19d6b40fca7d93 23019 100644 helicity_operator_comparison.py
6855771d8dc31cb8f39401c9957b6c962d52b5da4911a2d042fc0b26fd095346 92929 100644 helicity_three_operators_tau0p500_N231_z2_midpoint_P345.png
6de783125a097439f4767c525bf1336f30dbf4bbbd90fd8fd30b3be05e468b2e 285390 100644 helicity_three_operators_tau0p500_N231_z2_midpoint_real_imag_tsep5to11_P345.png
a23b81b1f7efd4389a062eb7c9c5e04e59640308bff70e671023b983c3cef8da 109853 100644 helicity_three_operators_tau0p500_N231_z2_midpoint_tsep5to11_P345.png
270fa4751219d1ea120746995f8c75405ec5981b57f4c097366e050612a385fe 217565 100644 helicity_three_operators_tau0p500_N231_z2_tsep12_P345.png
d8604e51942489b3f7dd859832ed37e6d08838799b3e498f6ff4faafecefe388 267449 100644 helicity_three_operators_tau0p500_N231_z2_tsep8_P345.png
980f50bbf8b96c3b373c64939559ef18158e6d87e6a4bcd52cb0ce7fb40caf60 269432 100644 helicity_three_operators_tau0p500_N231_z2_tsep8_imag_P345.png
efdcb456bdf112b265c251d871c64ad305b64e993676683731652dd093f892ae 236777 100644 helicity_three_operators_tau0p500_N231_z2_tsep8_real_P345.png
21fa8579f217f11fa8ed4b9127f667341847ca96840a709df3182ef533c7ff3b 288 100644 insertion_kernels.py
353f8a66e5a2c8a69a5530ba7c9a28a17eacfef36ff07d6633108afdabacf227 3501 100644 inventory20.py
ce0ad4045dcf42fab6bc964ac03ff4e0d149e53b9d3afd2da109ee05b0a5f76b 2950 100644 inventory30.py
07f1d99408f54034f3bdffd7e07c57b24708d4afb8c92a49d90bbd36728b7d46 11656 100644 lambda_fourier_quasi_matching_v1.py
e57f7d05b10bf980207960817027b760bb170b91126baf431315cf2ffc8fe548 20785 100644 lambda_fourier_quasi_matching_v2.py
7dec16e026e8ac5c29370bd957f51d4319a388393ea3611fb71c5fa579330b16 19167 100644 lambda_fourier_quasi_no_z0_v1.py
7f5dd1eb2d902f8426d042dd8f814c0fcd89b1dbc59815a4d071dcc3638838c8 1327 100644 launch.py
0aca3f598017e06dffa5766de1a7d6b73c27281c37e8e549a1f8fb212cca4ec3 321937 100644 linear_extrap_tau_gt1_z2.png
c7a051707cae10bf2e21d2c8bfb02fc94c85fd1f3a8ec67dad3f74f19d1a2bfb 3055 100755 make_manifest.py
ef074441a55b56980c645ca43f70cb8bc57ca8aafd7c4679e421bbcf36d40066 4854 100644 mass_prescription_audit.md
eeed143239712af2f2584541f76bbf1adbf3e17d8ab4154c73f35b3e96085008 607 100755 match_all_tau.sh
1e421ce7a059e2e29b5938ad941b8b921e3ccc332d0bd7464dad4fa1e9656fd6 10781 100644 match_bare_all_operators_v1.py
ca3e5f14a57be5681ed250842932d29913fca0f76e7bd286a3761cd5e41b9912 10154 100644 match_unpolarized_quasi_from_extrap_v1.py
eb7e1cdddb0bd268aeba5060bfdae7ce6ae414c0b7f1538602f47064deb6a4b8 13342 100644 match_unpolarized_quasi_from_extrap_v2.py
ef1546187282843528bd8ee07b701e26d92b87082520a66988a616fc2efd4ca6 17302 100644 measure.py
139273213272d5a29b35106c6f56a355af60d558e09b36c978e3144c1bacbc97 1256 100644 memory_bounded.py
b80d4b527672eed8fd9800dab30f0f14d48984661edd247343ed9eec3615bbb5 14673 100644 ope_components.py
d2b1bca71e06f6fc046f6349b6c668a8d54ced4ca55afd13b869e600e10f267c 5628 100644 operations.md
43b89881658625503144f82200762c6ecc677f05d13bd5a514ba7ed8e545952b 2375 100644 pilot.py
e2bf653be9e330874686b6a78999ddf787f03427e6d2aa846bd7ffa6f6a0052d 19516 100644 pipeline.py
406700d61e99bfeeb33b4e29894fc6ff1faf4d381488f452dec277009d8380af 6455 100644 plot.py
1b6e80de2dd4c0f3b7f4e5ce7434260f532d643017dab8e0c2cd0454588958ff 6721 100644 plot_after_zchi.py
80dba56509eebe38f44dba05b9c572e16fe8e3ad7247d9aa19e681c35ac6c04b 7978 100644 plot_and_extrapolate.py
c842a0bec6a48a9cb31125374f5607975a2b6a3c54f09e09f3aa7f86edee4f28 20088 100644 plot_bare_matrix_elements.py
df07283aedf9f496fa2082fb11553403ea674e76cc0cff743ca2c08fbac88281 3688 100644 plot_bare_results.py
0f50269923219afe313b1f197dcd09a12de403adec03532fd42f7e2c86b896e4 16925 100644 plot_fig1_style.py
2be16898ab535b6d171624e7d4e9f9b3c1ca70424305bfd53a83580495bf15a9 11591 100644 plot_matched_bare_all_operators_v1.py
320f110e297ca8cc1335af7704bf2894c8044abee323f39b0e9f09be7f938c55 19036 100644 plot_matched_bare_matrix.py
818a38b57eda4159697bf7c1cce909935f170873328bce38165edda93204929a 15392 100644 plot_matched_selected_taus_by_pz_v1.py
ca6869585de7941573e12d0416df9acbb4fe0d8d5db103e100ee02039e7b65fb 12644 100644 plot_pz3_results.py
29eb6b4194109763060f8008d372866f341586cfd2425275db806b986625771b 8567 100644 plot_tau_gt_0p1.py
09f1c8bd0855426a8a8df6d577e116578cf25de656a191dca5833ecf2df83437 10256 100644 plot_unpolarized_multiflow_fixed_ylim_v1.py
43f8767c7f76b217993c48fc42b4f3a8f1f786afd7c21d6e2a19173d34681392 3898 100644 prepare.py
41e43b1e229834934bafcda2421319bd7739de2290605a1327b2e152a65e8451 3191 100644 prepare_cpu.py
d880c42a9d467b390c7292214383a44c7b330a03395ca0ee42add763616b0248 2860 100644 prepare_suite.py
a72291febafc6019287d0370ad5baeecfe56fe086dd71567cfb646714faf12f2 3286 100644 quick_smoke_cpu.py
8bdcbc39802e8a839e7f7152c4f96f2a0862f3834a6c2e02f078f6365fb4f05d 7303 100644 ratios_cpu.py
ea900bb98b596f186a92f66dcbb356c5b1c47fd6602f257cda38028c12b58b43 1540 100644 raw_io.py
aa603b1a6ae71ba3d1ad40861204c3eebba2995eb04076e35564a62b0509178f 1135 100644 record_parallel_start.py
d8b60639851b4222365be5623ec8e619cb1302b18bdb3bcd3403396d02a97cd2 4712 100644 refine_no_z0_quasi_support_v1.py
d09209884e00bd4b6df813164c889c90dc8e94ee7240ee583ec052bdbdc4058c 17977 100644 relaxed_unpolarized_flow_extrap_v3.py
97b981691779f21611af23321bea59e578aed793442901824f4ca55e1e30d367 5711 100644 replica_selection_minerror_v3.py
ef536576119c283e0fed59d56cdda7d807d3c29bbf245f5e9068392039c08a5d 5444 100644 replot_unpolarized_flow_extrap_v2.py
6080027519ceded35c172230fe357fe4b5c218423d93088cb8681e5ac5513488 6625 100644 report_gpu.py
3d6e4381f3e0a27f783eeb2e412e77c121d3b3864c7a6d6d48ea28e27fefc19c 606 100644 require_regression.py
ddaeaef471bf05abee29077489c969898011ff172a1b53eb87e10ff954ba507e 9302 100644 resume_spatial_v5.py
10e017fe0714297bea4998944b25968eb394c760c9eab9e8bc32ad14d1c2d0d9 12390 100644 review.py
73456939584e9662a1993457e6b062ff669c3ed43772b42967312395691b43c6 3997 100644 review_recovery.py
a49b875cc021f6a6639ef576b7b0fd94bd0d52956402b0ef06e4f156a455534e 3469 100644 run_B_from_cache.py
744c5ac1e349015cf12cfe522b34d3b057b16db39313a4d0728b6fc982a30e23 670 100644 run_gpu.sh
910ea15e84a1fb45b579bc3cf50b08852139b1f83da6dba8403ae0331eaf2f37 3288 100644 run_memory_pipeline.py
1a9ac2b95619a50dfc753eb257ffb11a97c717059182adee870ae4ccb91d33fe 2514 100755 run_one.py
7864ed7a450590a0e179fa0111e660d2de2534cb1200bcd43461f6d3fb3ab9fa 2855 100755 run_one_v5.py
e0f8f173f07f5a9a361a75dc1dee650b615febb7ef7f1c893de0b7a7d4b251bf 6018 100644 run_pipeline.py
5794d501269aa811145794988bea1c7cf96ec5933e11a9c1310b20fdf911ddcd 25322 100644 run_quark_bare_20.py
2d21eb679a1d21e6bd1f0425172f31604d64569158fc974276e5c22b8da0359e 4963 100644 run_spin2.py
8b2c1fffe2ed3ba218eca83a3c4e42fdf7ef54b3a821385743ef15d5b765bd3d 3783 100644 run_task.py
03b128c7846cbd6f649674ba9255918d63a4702aa9dd25aa31113df26963c927 380 100755 run_tests.sh
7272cbaf8548a6af55ee689296ba04b4bc0d356e439fefe2b60bbc2124893407 6085 100644 spin2_contractions.py
063f0f0dd6e52e920cfd5a62326acc27212a56cf3db6b299008bb07f8c7c7480 2510 100644 stage_submit.py
415b481b641ab6b754580fce0a185a24d9b2439b24e9cb9d895b365cc00e1a2e 2527 100644 submit.py
00181a51775d5251e762f4d6474a9131cf17b36173116512e3327cd124aa409f 482 100644 submit.sh
3aca195e1d794911da97c6bd4076c5d0a6ae7617f7f2b94cd547721426cad2ab 2238 100644 submit_bundled_recovery.py
18d660cbfbdfc9a3a51bc6ad45da61138bdc69e2fa9b4109ca488d9c2e3b5396 3047 100644 submit_cpu.py
1711343fae2bab8921b73bf5f8dc56f36e70298b062f0272d8593ab839bee19c 2834 100755 submit_cpu.sh
ab753c6229ad00e3390d3c0b83ba1bb0f7b786e47283ba74c4bfe61f9ae1b9f7 2859 100644 submit_fill_flows.py
3bb2cd84fb09163cfcb53165b632bb0913b7ed8f5ce6bf851e3c8a75404c0a53 506 100644 submit_missing_retry_20261006.sh
3ef5bac4640b82e735e5b4e4ba78d0e28d8501b7c0343ab2c490ad43c7e5267e 1972 100644 submit_new10.py
ea0ce9d7efe02111759cd939c340a31f11c3f27b7934f537e7afc0e830f44ca2 1905 100644 submit_production.py
1bdc930a0b9782dcd21453ec880d3d1ac6d134277b87d9982d2fa8151c43d23c 1845 100644 submit_production_parallel.py
1b9bf16ccb5fb8f77b3f67723365de273e32910e9f204015185b2007094132a1 562 100755 submit_tf03_tf04.sh
72044a660dfe9faa81dcf1abd86abb2a45ef8b71f6e53660316203366fab55c0 627 100644 submitted_005_01_20260913_151237.md
5cdf1741a04347db3cc8f49284a6c67e2daf4344ac73cb8179815c9cd445452e 6008 100644 summarize.py
33677c1177e3b3beb393f0fef05907afe2843d118f86f42e15e926fcd297e76f 6394 100644 summary_cpu.py
4726b3b91743e3d5922ea4384a53f66221b53e6bb5a80f66d5c65a980700b968 1881 100644 test_analysis.py
e36cfd09c73151de682c754f32d2662b97330d7ad58ccb0e3b43b89a1d342174 7395 100644 test_components.py
b44891e625ede128319e2df70fd621e674f0a4a0e02466944df6bc7f11c9c00d 4596 100644 test_cpu.py
18e10db9b3c17fbb66d4ea467e63a08372be59e170654e55e675f5a21a17875d 4966 100644 test_driver.py
b9781c78e559c209165acae5854cf564054cbca2a8c948e527c0bdb8a8e4a409 3847 100644 test_extrapolate_unpolarized_tau_ge1_correlated_v2.py
7019069ce3c00f65cdfbeffb1040300ac09834ecee1c4e09759d73e64d11a705 1053 100644 test_gluon_flow_to_quasi.py
bc691167f4f71d2fa1b28c82ed958e10f8d8575418bc08366bbb857a1ef5751e 3229 100644 test_helicity_operator_comparison.py
2f8752f157e0858c51733084bdd0edb3b3413fdf1f55ff89e078838575fac3b9 3719 100644 test_insertions.py
4e2942d98d7571d2f3dd8cadcbda90086bcbfdf93c6d17eb533e8ceb75bea4ce 11863 100644 test_kinetic.py
34c935bf425ff7d5fe5682b4438ff017c04becfdb49e220887dc7c4b81bfbfbd 3147 100644 test_lambda_fourier_quasi_matching_v2.py
bc5fa66d84705139ab3b14854d8d21a4e0bab9c76570949e714fc0d1beb9e130 3507 100644 test_lambda_fourier_quasi_no_z0_v1.py
380d8bcc56676cc517edc2ac6300b625b09575a087ec0e073453834869311142 1306 100644 test_multidirection.py
727a1ee47381c8e105b0cddf94173f3c4033516f11b8875dbcec2eb4feb426d5 1331 100644 test_pipeline.py
bd07821e489db3f540eb1975b0ff5b07b59397a6eb6ed2d34cca79133d8f9c43 3255 100644 test_replica_selection_minerror_v3.py
9988d7462de78e442b89a07dd9d435033f5f5aab30f77957ed20b25d72eb620b 7342 100644 test_small.py
0519c5038f9d6d69872adf2dac10be8a36299c8a33511c4b94c5fe1e7486b440 2510 100644 test_spin2.py
2a40ecd014c3c7eec35bf5215c36930df3b204f8abc2d085573ba7209063304b 237499 100644 unpolarized_u_minus_d_sixflow_bare_matrix_elements.png
850286a4b2d8c60d29734b64db5bbbf6fac0e8f4a922d08b0b3bb08847a0636a 466 100644 validate_and_prepare.py
f645c55e214d76af9fb18821d288a733f960d7d7bc7cd603fc4d11eeffa590b8 8336 100644 validate_collection.py
4ee06eea49ab0d377ce5347356f30b840651e13413a0f1bc90bfd9ef5946d9dc 5143 100644 validate_lambda_fourier_quasi_matching_v2.py
3bfbc80bbe733e9024cd4e74ceb71d31282d2a0bfbe1b3e48146bd3a7846d2fd 4145 100644 validate_lambda_fourier_quasi_no_z0_v1.py
ca11066bb2ca9a3a905e005013723cdcefcd30955888a926dce1928167a7d06b 2723 100755 validate_output.py
3a7e6d0645aa946444e5984f682100fe7b5713d7bde62df92ea53c542c0e20cc 4330 100755 validate_output_v5.py
9a8c55b6d59f5dae8856c5985b8582836b63af43d468ba5114885a7186849716 3663 100644 validate_product.py
51d39409a9f695704adad2baf532cb6f32e6f97c783e18854d3fc4b5c5efda1d 1602 100644 validate_products.py
3c53e921fc596e41f1c1dfefca370e2cb0115dd6d0a9739b7b6963caeb2e66c2 4961 100644 validate_relaxed_unpolarized_v3.py
3ff48fe92c25d1de0d78be0b3b017e4344250b035bcb3c66403aa0c2a06a9b4f 4383 100644 validate_results.py
1ea52e7aabe9bd3a521173fc540dd41bac74d5e1e460ff31c537ad38f65a81f3 2248 100644 verify_cleanup_retry.py
da1343456ec47230eb99a1ae71a82c6126ea92fdcdc455750a8e9e4390269b53 3573 100644 weights_and_terms.py
3d2d2ce423ee470e4b2cee032668f7a1e37d2ff4a50d668eb9ea8949a936042e 1009 100644 worker.py
863594fa67601422f321c7e2439fd12fb8e0c7a6c4312bb24d820751454b908a 1193 100644 write_product_index.py
```

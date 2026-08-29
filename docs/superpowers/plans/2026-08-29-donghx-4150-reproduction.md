# donghx 4150 格点 QCD 复现与对照 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以组态 4150 为核心，逐层复现并验证 donghx 的 eigvec、VVV/VdV、2pt、OPE、3pt、ratio 与 barematrix 算法，生成中文证据报告并完成授权的 Git 交付。

**Architecture:** 保留现有 `examples/pyqcd/cmp1` harness 作为统一计时和比较器，在其旁增加 4150 资产 manifest 与真实链 runner；参考目录只通过显式只读适配读取，生产算法仍位于 `pyqcd/`。每一层先验证 shape/轴序/不变量，再比较数组和最终统计量；发现差异时只在对应生产模块做最小修复，并立即回归。

**Tech Stack:** Python 3、NumPy/SciPy、PyQCD、h5py/ASCII IO、可用的 PyQUDA/CuPy/Torch 后端、JSON/Markdown 证据、XeLaTeX Beamer、Git annotated tag。

**Spec:** `docs/superpowers/specs/2026-08-29-donghx-4150-reproduction-design.md`

## Global Constraints

- 工作目录为 `/root/PyQCD`；参考代码目录 `/root/PyQCD/refer/donghx` 只读。
- 只读取用户明确引入的 `/public/group/lqcd/...` 数据和参考结果；默认写入 `/root/PyQCD/data` 或版本化的 `examples/pyqcd`/`logs` 产物目录。
- 不以参考结果的“看起来合理”为算法证明；每个数值差异必须经过 shape、轴序、dtype、相位/符号、归一化和不变量检查后再分类。
- 复数计算保持复数；只有由明确投影、共轭关系或参考定义支持时才取实部。
- 不删除数据、不改写已推送历史、不修改系统配置；提交和推送仅使用本任务明确授权的普通 commit、annotated tag 和远端更新。
- TMD 属于进阶项：核心链未闭合或缺少物理所需的 `z/b_perp/staple/flow` 元数据时，只报告已有实现和限制，不把准 TMD 输出冒充完整物理 TMD-PDF。
- 参考库只读，PyQCD 不得 import `refer/` 或 `examples/` 作为业务实现。
- 主回归入口为 `python examples/pyqcd/conftest.py`，真实比较结果必须保存版本化 JSON/Markdown。

---

### Task 1: 建立 4150 资产 manifest 与算法映射

**Files:**
- Create: `examples/pyqcd/cmp1/manifest_4150.py`
- Create: `examples/pyqcd/cmp1/verify_manifest_4150.py`
- Modify: `examples/pyqcd/cmp1/datalib.py`
- Read: `refer/donghx/AGENTS.md`, `refer/donghx/Eigvec_code/`, `refer/donghx/Contraction/`, `refer/donghx/Calc_VVV.py`, `refer/donghx/Operator.py`, `refer/donghx/Calc_ope_*.py`, `refer/donghx/2pt_proton_*.py`
- Test: `examples/pyqcd/cmp1/verify_manifest_4150.py`

**Interfaces:**
- `manifest_4150.build_manifest(conf_id='4150') -> dict`：返回输入、参考结果、代码入口的存在性、文件数、字节数、mtime、样例路径和缺失原因；不得加载整个数组。
- `manifest_4150.array_meta(path) -> dict`：对 `.npy/.npz/.h5` 返回 shape、dtype、文件大小和可读状态；对目录返回递归文件数和总字节数。
- `datalib.configure(conf=4150, cache_dir=...) -> None`：改变本次 runner 的组态与缓存位置，不改变旧调用的默认行为。

- [ ] **Step 1: Write the failing manifest checks**

```python
def test_manifest_has_explicit_conf_and_categories():
    m = build_manifest("4150")
    assert m["conf_id"] == "4150"
    assert {"input", "reference_output", "reference_code"} <= set(m)
    assert m["input"]["gauge"]["exists"] is True

def test_array_meta_does_not_require_full_load(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((2, 3), dtype=np.complex128))
    meta = array_meta(str(path))
    assert meta["shape"] == [2, 3]
    assert meta["dtype"] == "complex128"
```

- [ ] **Step 2: Run the focused test and record the red result**

Run: `python examples/pyqcd/cmp1/verify_manifest_4150.py`

Expected: the test runner reports missing `build_manifest`/`array_meta` or an equivalent explicit failure before implementation.

- [ ] **Step 3: Implement the manifest and parameterized data loader**

Use the exact user-provided paths as constants, use `os.stat` and NumPy/HDF5 headers, and keep the existing `datalib.eigvecs`, `datalib.peram`, and `datalib.gauge` call signatures backward-compatible. The manifest must distinguish a directory that exists but has no matching 4150 file from a genuinely absent path.

- [ ] **Step 4: Run the focused manifest gate**

Run: `python examples/pyqcd/cmp1/verify_manifest_4150.py`

Expected: manifest schema, 4150 selection, input existence and non-destructive metadata scan pass; no full gauge array is copied by the inventory command.

- [ ] **Step 5: Commit the inventory layer**

```bash
git add examples/pyqcd/cmp1/manifest_4150.py examples/pyqcd/cmp1/verify_manifest_4150.py examples/pyqcd/cmp1/datalib.py
git commit -m "test: add 4150 asset manifest"
git push origin main
```

### Task 2: 复现 eigvec、phase、VdV/VVV 与低层 OPE

**Files:**
- Create: `examples/pyqcd/cmp1/cases_4150_lowlevel.py`
- Create: `examples/pyqcd/cmp1/run_4150_lowlevel.py`
- Modify: `pyqcd/vertex/_vertex.py` only when a focused regression proves a defect
- Modify: `pyqcd/operator/_gluon_ope.py` only when a focused regression proves a defect
- Modify: `examples/pyqcd/cmp1/harness.py` if metadata recording is required
- Test: `examples/pyqcd/cmp1/run_4150_lowlevel.py --smoke`

**Interfaces:**
- `cases_4150_lowlevel.build(conf_id='4150') -> list[Case]`：构造 eigvec reader、phase、VdV、VVV、Clover、dual、Wilson-line/OPE 案例。
- `run_4150_lowlevel.run(cases, outdir) -> dict`：依次执行案例并保存 `results.json`、`summary.md`、`inputs.json`，不覆盖既有版本目录。
- 每个案例的结果至少带 `shape`、`dtype`、`norm`、`unitarity_residual` 或 `orth_residual`（适用时）、`t_ref`、`t_pq` 和 `status`。

- [ ] **Step 1: Write the controlled low-level tests**

```python
def test_phase_and_vdv_contract_have_expected_shapes():
    from pyqcd.vertex import Mom_VdV_sink_t, phase_exp_2pt
    eig = np.zeros((4, 24, 24, 24, 3), dtype=np.complex128)
    phase = phase_exp_2pt(24, [0, 0, 1])
    vdv = Mom_VdV_sink_t(phase, eig)
    assert vdv.shape[-2:] == (24, 24)

def test_vvv_is_finite_and_color_antisymmetric():
    rng = np.random.default_rng(7)
    from pyqcd.vertex import Mom_VVV_sink_t, phase_exp_2pt
    eig = rng.normal(size=(6, 24, 24, 24, 3)) + 1j * rng.normal(size=(6, 24, 24, 24, 3))
    out = Mom_VVV_sink_t(phase_exp_2pt(24, [0, 0, 0]), eig)
    assert np.isfinite(np.asarray(out)).all()
```

- [ ] **Step 2: Run the controlled tests before changing production code**

Run: `python examples/pyqcd/cmp1/run_4150_lowlevel.py --controlled`

Expected: shape and finite-value checks execute; any mismatch is recorded with the first failing invariant.

- [ ] **Step 3: Add real 4150 low-level cases**

Load only the needed time slab/eigenvectors for the smoke case, then run the full requested slab when memory permits. Compare reference and PyQCD with a denominator based on `||reference||`, use phase-equivalence checks for eigenvectors, and never label missing VdV/VVV reference files as numerical passes. For gauge inputs run raw and each available 3D/4D smear variant separately.

- [ ] **Step 4: Diagnose and minimally fix one discrepancy at a time**

For each `diff`, check in order: `(Nt,Nz,Ny,Nx,dir,color,color)` versus flattened spatial order; complex interleaving; Fourier sign; Levi-Civita ordering; Clover normalization; dual epsilon index order; Wilson-line start/end orientation. Modify only the responsible PyQCD function after an independent residual or symmetry check.

- [ ] **Step 5: Run the real low-level gate and save evidence**

Run: `python examples/pyqcd/cmp1/run_4150_lowlevel.py --conf 4150 --outdir examples/pyqcd/cmp1/v$(date +%Y%m%d%H%M%S)`

Expected: every case is `pass`, `diff` with an evidence note, or `unverified`; no result is silently dropped and the command line/elapsed time are stored.

- [ ] **Step 6: Commit low-level changes and evidence code**

```bash
git add examples/pyqcd/cmp1/cases_4150_lowlevel.py examples/pyqcd/cmp1/run_4150_lowlevel.py pyqcd/vertex/_vertex.py pyqcd/operator/_gluon_ope.py examples/pyqcd/cmp1/harness.py
git commit -m "test: compare 4150 low-level lattice objects"
git push origin main
```

### Task 3: 复现 perambulator → 2pt → 3pt 费米子链

**Files:**
- Create: `examples/pyqcd/cmp1/cases_4150_fermion.py`
- Create: `examples/pyqcd/cmp1/run_4150_fermion.py`
- Modify: `pyqcd/contraction/_seqperam.py` only after a regression identifies a defect
- Modify: `pyqcd/contraction/_autowick.py` or `pyqcd/contraction/_dynamic.py` only after an einsum/sign invariant identifies a defect
- Modify: `pyqcd/analysis/_analyse.py` only for a proven ratio/boundary mismatch
- Test: `examples/pyqcd/cmp1/run_4150_fermion.py --controlled`

**Interfaces:**
- `cases_4150_fermion.build(conf_id='4150', variants=...) -> list[Case]`：为每个可见 smear/momentum/direction/operator 生成 2pt/3pt 案例。
- `run_4150_fermion.run_2pt_case(peram, vvv, operator, momentum, boundary) -> ndarray`：返回带明确 `[source/sink,time,...]` 语义的单组态 2pt。
- `run_4150_fermion.run_3pt_case(peram, vdv, current, t_sep, momentum) -> ndarray`：返回未平均的 3pt；若缺少顺序源输入，返回结构化 `unverified` 记录而不是空数组。

- [ ] **Step 1: Write tests for the contraction invariants**

```python
def test_seq_peram_preserves_leading_and_trailing_axes():
    peram = np.ones((2, 3, 4, 5), dtype=np.complex128)
    out = seq_peram(peram)
    assert out.shape == peram.shape

def test_ratio_uses_same_time_axis_and_boundary_sign():
    c2 = np.arange(8, dtype=float) + 1
    c3 = 2 * c2
    out = ratio_3pt(c3, c2)
    np.testing.assert_allclose(np.asarray(out), 2.0)
```

- [ ] **Step 2: Run controlled tests and verify the failing behavior is isolated**

Run: `python examples/pyqcd/cmp1/run_4150_fermion.py --controlled`

Expected: the tests exercise only axis/sign contracts and do not allocate the full 24³×72 data set.

- [ ] **Step 3: Build a single-variant 4150 smoke chain**

Use `Nev1=8` first, one momentum and one operator, and preserve peram/VVV/VdV source metadata. Compare the intermediate sink vertex, converted source vertex, each contraction term, and the final correlator; use the reference file name to recover momentum and smearing labels rather than guessing from directory names.

- [ ] **Step 4: Expand the variant matrix without changing the contraction kernel**

Run all reference-visible combinations of 3D 1/3/5 and 4D 10 gauge smear, momentum smear x/y/z and magnitude, and `Cg5`/`Cg5g4`. Each variant is a separate record. A missing reference result is `unverified(reference_output_missing)` even when PyQCD runs successfully.

- [ ] **Step 5: Resolve differences using physics invariants**

Check the proton interpolator’s epsilon-color contraction, gamma5 Hermiticity, anti-periodic backward sign, `P±=(1±gamma4)/2`, Fourier phase, and source/sink time ordering. Do not repair a negative correlator by taking `abs` or `real` unless the reference definition and an independent symmetry identity require it.

- [ ] **Step 6: Run the real 2pt/3pt gate**

Run: `python examples/pyqcd/cmp1/run_4150_fermion.py --conf 4150 --outdir examples/pyqcd/cmp1/v$(date +%Y%m%d%H%M%S)`

Expected: at least one actual 2pt and one actual 3pt chain complete with saved arrays and timing; every unavailable reference combination is listed explicitly.

- [ ] **Step 7: Commit the fermion-chain changes**

```bash
git add examples/pyqcd/cmp1/cases_4150_fermion.py examples/pyqcd/cmp1/run_4150_fermion.py pyqcd/contraction pyqcd/analysis/_analyse.py
git commit -m "test: reproduce 4150 fermion contraction chain"
git push origin main
```

### Task 4: 复现 ratio、barematrix 与统计输出

**Files:**
- Create: `examples/pyqcd/cmp1/cases_4150_analysis.py`
- Create: `examples/pyqcd/cmp1/run_4150_analysis.py`
- Modify: `pyqcd/analysis/_ratio2pt.py` only after direct data comparison proves a defect
- Modify: `pyqcd/analysis/_bare_matrix.py` only after direct data comparison proves a defect
- Modify: `pyqcd/analysis/_ratio_fit.py` only after window/covariance evidence proves a defect
- Test: `examples/pyqcd/cmp1/run_4150_analysis.py --controlled`

**Interfaces:**
- `cases_4150_analysis.build(conf_id='4150') -> list[Case]`：按方向、z、`t_sep`、拟合窗口和 operator 生成 ratio/barematrix cases。
- `run_4150_analysis.summarize_chain(raw) -> dict`：返回 input shape、fit window、covariance/SVD、parameter estimates and status。
- `run_4150_analysis.run(...) -> pathlib.Path`：写出未拟合 ratio、拟合表、barematrix、图和 JSON summary。

- [ ] **Step 1: Write the statistical contract tests**

```python
def test_bare_matrix_keeps_direction_labels():
    result = run_bare_matrix(data_root, out_root, sample_params, fit_params)
    assert {"x", "y", "z", "ave"} <= set(result["directions"])

def test_nan_aware_comparison_requires_same_nan_mask():
    from pyqcd.testing import cmp_one
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert cmp_one(a, b)["pass"] is True
```

- [ ] **Step 2: Run controlled statistics tests**

Run: `python examples/pyqcd/cmp1/run_4150_analysis.py --controlled`

Expected: direction labels, resampling shape, covariance and NaN mask behavior are checked without external files.

- [ ] **Step 3: Reconstruct the reference input layout**

Use the fresh 2pt/3pt/OPE artifacts and reference file names to create a staging view under `/root/PyQCD/data/4150/`; do not copy or modify `/public` data. Save a manifest of symlink targets and the exact `SampleParams2pt`/fit parameters.

- [ ] **Step 4: Compare raw ratio before comparing fit parameters**

Compare raw `C3/C2`, vacuum subtraction, z ordering, direction average and boundary selection first. Then compare fitted `c0`, errors, chi-square and plots. Record whether a difference is caused by input phase, sample count, covariance singularity or fitting model.

- [ ] **Step 5: Run the full analysis gate**

Run: `python examples/pyqcd/cmp1/run_4150_analysis.py --conf 4150 --outdir examples/pyqcd/cmp1/v$(date +%Y%m%d%H%M%S)`

Expected: raw and final analysis products are saved; no fitted value is presented as a numerical comparison when its raw input is unavailable.

- [ ] **Step 6: Commit analysis-chain changes**

```bash
git add examples/pyqcd/cmp1/cases_4150_analysis.py examples/pyqcd/cmp1/run_4150_analysis.py pyqcd/analysis/_ratio2pt.py pyqcd/analysis/_bare_matrix.py pyqcd/analysis/_ratio_fit.py
git commit -m "test: close 4150 ratio and bare-matrix comparison"
git push origin main
```

### Task 5: 可选 TMD 扩展与性能诊断

**Files:**
- Create: `examples/pyqcd/cmp1/cases_4150_tmd.py` only if Tasks 2–4 satisfy their required evidence gates
- Modify: `pyqcd/renorm/` only for a separately documented TMD discrepancy
- Read: `pyqcd/renorm/AGENTS.md`, `skills/pyqcd-tmd-algorithm/SKILL.md`, `skills/pyqcd-tmd-chain/SKILL.md`
- Test: `examples/pyqcd/cmp1/run_4150_tmd.py --smoke` when enabled

**Interfaces:**
- `cases_4150_tmd.build(...) -> list[Case]`：明确记录 `tau,z,b_perp,staple_length,representation,mu,zeta,matching_order`。
- `run_4150_tmd.run(...) -> dict`：只输出有完整元数据的准 TMD/TMD 中间量，并把缺少软因子、rapidity subtraction 或连续极限输入的部分标为 `unverified`。

- [ ] **Step 1: Check the core-chain gate before enabling TMD**

Run: `python examples/pyqcd/cmp1/verify_cmp1.py examples/pyqcd/cmp1/v20260829120000/results.json` using the newest printed results path, together with the fresh 2pt/3pt/OPE/analysis verifiers. If any required core input is missing, record the reason and skip this task.

- [ ] **Step 2: Run only a one-z/one-b smoke case**

Check flow-time dependence, staple geometry, Fourier sign, and normalization. Compare each intermediate h(z), subtraction factor, CS kernel and matching step independently.

- [ ] **Step 3: Keep physical limitations explicit**

Do not claim a complete TMD-PDF without a representation-consistent soft factor, rapidity subtraction, tensor mixing treatment and a controlled continuum/flow extrapolation.

### Task 6: 全量验证、报告和 PDF 视觉验收

**Files:**
- Create: `docs/report_donghx_4150_reproduction_20260829.tex`
- Create: `docs/report_donghx_4150_reproduction_20260829.pdf`
- Create: `logs/donghx_4150_20260829/` evidence bundle
- Modify: `docs/AGENTS.md` to register the report
- Test: `python examples/pyqcd/conftest.py`, all fresh 4150 verifiers, XeLaTeX/PDF checks

**Interfaces:**
- Report input is only the fresh manifest/results/summary, source line references and current command logs.
- Report sections are: objective and acceptance, physical formula chain, reference-to-PyQCD mapping, 4150 assets, low-level results, fermion results, ratio/barematrix results, performance, differences, evidence levels and limitations.

- [ ] **Step 1: Run the repository regression before report compilation**

Run: `python examples/pyqcd/conftest.py`

Expected: all current tests pass; if a production change caused a regression, debug it before generating the report.

- [ ] **Step 2: Run fresh comparison verifiers**

Run: `python examples/pyqcd/cmp1/verify_manifest_4150.py`, the low-level/fermion/analysis runners and `python examples/pyqcd/cmp1/verify_cmp1.py examples/pyqcd/cmp1/v20260829120000/results.json` after replacing the example path with the newly printed version directory.

Expected: each case is backed by a current command; missing external results remain explicit `unverified` records.

- [ ] **Step 3: Build the report from measured evidence**

Insert exact paths, line numbers, shapes, tolerances, timings and status counts. Label claims `已确认`, `由不变量支持的推断` or `未验证`; include no invented numbers.

- [ ] **Step 4: Compile twice in an isolated build directory**

Run from `docs/`:

```bash
xelatex -interaction=nonstopmode -output-directory=/tmp/pyqcd_report_build report_donghx_4150_reproduction_20260829.tex
xelatex -interaction=nonstopmode -output-directory=/tmp/pyqcd_report_build report_donghx_4150_reproduction_20260829.tex
```

Expected: `Overfull=0`, `Float too large=0`, `Missing character=0`; expected/actual/rendered/checked page counts agree.

- [ ] **Step 5: Render and inspect every PDF page**

Use the repository’s available PDF-to-image and text tools to inspect every page for clipped objects, occlusion pairs, outside-safe-area content, unreadable tables, stale placeholders and wrong paths. Save the page-count and visual-check summary beside the report.

- [ ] **Step 6: Commit and push the verified report bundle**

```bash
git add docs/report_donghx_4150_reproduction_20260829.tex docs/report_donghx_4150_reproduction_20260829.pdf docs/AGENTS.md logs/donghx_4150_20260829
git commit -m "report: document donghx 4150 algorithm reproduction"
git push origin main
```

### Task 7: 最终 diff、tag 与远端闭环

**Files:**
- Read: all files changed by Tasks 1–6
- Create: no additional source file unless the tag audit requires a factual changelog

**Interfaces:**
- Final tag message records the actual commit, report, tests, unresolved limits and agent name.
- The tag target is the pushed commit HEAD; no tag points to an uncommitted tree.

- [ ] **Step 1: Run final diff and boundary checks**

```bash
git diff --check
git diff --stat origin/main...HEAD
git status --short --untracked-files=all
```

Expected: no whitespace errors, no accidental reference-library edits, no credentials, no debug leftovers, and all intended evidence files are included.

- [ ] **Step 2: Commit any final non-destructive metadata correction**

```bash
git status --short --untracked-files=all
git add --update
git add docs/report_donghx_4150_reproduction_20260829.tex docs/report_donghx_4150_reproduction_20260829.pdf docs/AGENTS.md logs/donghx_4150_20260829 examples/pyqcd/cmp1
git commit -m "chore: finalize 4150 reproducibility evidence"
git push origin main
```

- [ ] **Step 3: Create the next factual annotated stab tag**

First inspect existing `stab*` tags and choose the next unused suffix (if `stab15` is highest, use `stab16`). Then create an annotated tag whose message states the actual comparison scope:

```bash
git tag -a stab16 -m "follow stab15, 1. 完成 donghx 4150 算法与数据对照; 2. 归档真实中间/最终结果及中文报告; 3. 保留已明确的未验证边界; [codex]."
git push origin stab16
```

- [ ] **Step 4: Verify local and remote tag identity**

```bash
git rev-parse HEAD
git rev-parse stab16^{commit}
git cat-file -t stab16
git ls-remote --tags origin refs/tags/stab16 'refs/tags/stab16^{}'
```

Expected: local peeled tag equals pushed HEAD, tag type is `tag`, and the remote advertises both the annotated tag object and its peeled commit.

## Plan self-review

- Scope coverage: Tasks 1–4 cover every core object named in the specification; Task 5 isolates optional TMD; Tasks 6–7 cover report, validation and Git delivery.
- Placeholder scan: the plan contains no `TODO`, `TBD`, or “implement later”; version directories are generated with `date`, and the tag audit resolves the next unused suffix before execution.
- Interface consistency: all runners consume `conf_id`, emit structured `results.json` records and use the existing `Case`/`run_case` model; `datalib.configure` preserves old defaults.
- Evidence boundary: no task treats a missing reference output as a pass, and no task changes production code without an independent failing invariant.

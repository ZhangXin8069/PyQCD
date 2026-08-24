# 05_FH_bare_matele — FH 变换、拟合与画图

## 文件说明

| 文件 | 功能 | 运行方式 |
|------|------|----------|
| `1_preprocess_FH.py` | 加载 ratio → FH 变换 → 画 FH 图 | 登录节点直接运行 |
| `2_fit_FH.py` | 读取 FH → 做 fit → 保存结果 | 登录节点或 SLURM |
| `3_plot_FH.py` | 读取 fit 结果 → 合并/报告/画图 | 登录节点直接运行 |
| `submit.sh` | SLURM 作业脚本, 循环提交所有 z 的 fit | `sbatch submit.sh` |

> `code_FH_bare_matele.py` 已废弃, 功能已拆分到 `1_preprocess_FH.py`、`2_fit_FH.py`、`3_plot_FH.py`.

## 用法

### 1. 预处理 (登录节点)

```bash
python 1_preprocess_FH.py -c L24x72 -p 4
```

参数:
- `-c conf_short` (默认 `L24x72`)
- `-p P` (默认 `2`)

输出到 `01_result/{conf_short}/P{P}/fh/`:
- `FH_nex{N}.npy` — FH 变换结果
- `z{iz}.png` — FH 图 (多个 nex 对比)

### 2. 拟合 (SLURM 或登录节点)

**生产模式** (使用 prior, 单个 z, 全部样本):
```bash
python 2_fit_FH.py -c L24x72 -p 4 -z 3 -u
```

**调试模式** (使用 p0, 所有 z, 每个 z 前 100 个样本):
```bash
python 2_fit_FH.py -c L24x72 -p 4
```

参数:
- `-c conf_short` (默认 `L24x72`)
- `-p P` (默认 `2`)
- `-z Z` (可选, 指定 z 值; 不提供则进入调试模式)
- `-u` / `--use-prior` (可选, 使用 prior 拟合; 不提供则使用 p0)

输出到 `01_result/{conf_short}/P{P}/fit/para{model}_n{nex}_tsep{start}_{end}/`:
- 生产模式: `fit_z{Z}.npz` + `report_z{Z}.txt`
- 调试模式: `fit.npz` + `report.txt`

### 3. 提交所有 z 的拟合 (SLURM)

```bash
# 修改 submit.sh 中的 conf_short 和 P, 然后:
sbatch submit.sh
```

submit.sh 会自动:
- 从 conf_short 提取 z 的数量 (如 L24x72 → 24 个 z: 0..23)
- for 循环顺序执行每个 z 的 `2_fit_FH.py`
- 每个 z 独立启动 Python 进程, lsqfit 每次重新初始化, 避免长时间运行变慢

### 4. 画图 (登录节点, 等待所有 fit 完成后)

```bash
python 3_plot_FH.py -c L24x72 -p 4
```

输出:

```
01_result/{conf_short}/P{P}/fit/merged/{tag}/
    fit.npz              # 合并后的拟合数据
    report.txt           # 报告
    pic_{param}.png      # 参数 vs z 图
    pic_chi2.png         # chi2/dof 散点图
01_result/{conf_short}/P{P}/pic/
    pic_{param}.png      # 各窗口参数对比图 (多组数据叠加)
    pic_chi2.png         # chi2/dof 对比散点图
    pic_bestfit_z{Z}.png # bestfit FH + c0 色带图
```

## 完整流程

```bash
# 1. 预处理 (登录节点)
python 1_preprocess_FH.py -c L24x72 -p 4

# 2. 提交 fit (SLURM)
sbatch submit.sh

# 3. 等待 fit 完成后, 画图 (登录节点)
python 3_plot_FH.py -c L24x72 -p 4
```

## 配置说明

### 拟合参数 (`2_fit_FH.py`)

- `_prior`: 全局先验参数 (gvar 格式), 在 `init_config()` 开头定义
- `_fit_params_by_P`: P 相关的拟合参数 (model, nex, dt_start_list, dt_end)
- `-u` 控制使用 prior 还是 p0

### 画图参数 (`3_plot_FH.py`)

- `_bestfit_by_P`: bestfit 窗口配置, 列表格式 `[model, dt_start, dt_end, nex]`
- `_cmp_dt_start_list`: 对比图的 dt_start 列表
- `_cmp_cfg`: 对比图画图配置 (坐标范围, 步长等)
- `_fh_cfg`: FH 图画图配置 (坐标范围)

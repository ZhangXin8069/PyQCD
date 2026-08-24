# 胶子 unpolarized PDF ratio 计算

## 功能

加载 OPE（Wilson line 算符）和 2pt（两点关联函数）数据，计算 disconnected 3pt 与 2pt 的比值 ratio，并对 6 个方向（pos_x, neg_x, pos_y, neg_y, pos_z, neg_z）分别保存 ratio 数据，最后画图。

## 输入数据

### OPE 数据

路径格式：
```
/public/group/lqcd/donghx/Ope_Gluon/Result_hpy_4D_10times/{conf_short}/{axis}dir/{conf_id}/ops_mu{tdir1}_nu{tdir2}_dz{Nx}_conf{conf_id}.npz
```

每个 `.npz` 文件包含 key `"ops"`，shape = `(Nx, Nt)`，为复数数组。

加载 3 个文件（`ops_mu{tdir1}_nu{tdir2}`、`ops_mu3_nu{tdir1}`、`ops_mu3_nu{tdir2}`）后组合为：
```
_ope = -_ope_ti - _ope_tj + 2 * _ope_ij
```

最终 shape = `(Nconf, Nt, Nx)`。

### 2pt 数据

路径格式：
```
/public/group/lqcd/donghx/2pt_Result/{conf_name}/momsmear{momP}{axis}/{conf_id}/twopt_slice_pp_Px{Px}Py{Py}Pz{Pz}_eginphase{momP}_Cg5g4_nopol_ss_conf{conf_id}.npy
```

shape = `(Nt, Nt)`，为复数数组。加载后 shape = `(Nconf, Nt, Nt)`。

## 输出数据

### ratio 数据

保存为 `ratio.npy`，分别保存在 6 个方向子目录中：

```
{result_dir}/pos_x/ratio.npy
{result_dir}/neg_x/ratio.npy
{result_dir}/pos_y/ratio.npy
{result_dir}/neg_y/ratio.npy
{result_dir}/pos_z/ratio.npy
{result_dir}/neg_z/ratio.npy
```

shape = `(Nsample, dt_max, dt_max, Nx)`

| axis | 含义 |
|------|------|
| 0 | resample 样本（jackknife 或 bootstrap） |
| 1 | tsep（源-汇时间间隔） |
| 2 | tins（插入时间） |
| 3 | z（空间分离） |

### 图片

每张图对应一个 z，保存在对应方向目录中：

```
{result_dir}/{dir}/ratio_{dir}_z{z}.png
```

## 使用方法

### 完整运行（计算 ratio + 画图）

```bash
python code_02_ratio.py -c L24x72 -s 1 -e 2
```

### 只计算 ratio

```bash
python code_02_ratio.py -c L24x72 -s 1 -e 1
```

### 只画图（需要已有 ratio.npy）

```bash
python code_02_ratio.py -c L24x72 -s 2 -e 2
```

## 路径说明

```
{result_dir} = ./1_result/{conf_short}/P{P}
```

| 变量 | 含义 |
|------|------|
| `{conf_short}` | 配置简称，如 `L24x72` |
| `{P}` | 动量大小 |
| `{axis}` | 方向，`x`/`y`/`z` |
| `{dir}` | 方向，`pos_x`/`neg_x`/`pos_y`/`neg_y`/`pos_z`/`neg_z`/`ave` |
| `{conf_id}` | 组态编号 |
| `{momP}` | 动量符号，`±2` |
| `{tdir1/tdir2}` | 方向索引，`0(x)/1(y)/2(z)/3(t)` |

## 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-c` | `L24x72` | 配置简称 |
| `-s` | `1` | 起始步骤：`1`=计算 ratio，`2`=画图 |
| `-e` | `2` | 结束步骤：`1`=计算 ratio，`2`=画图 |

### 代码内参数（在 `init_config()` 中修改）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `P` | `4` | 动量大小 |
| `momP` | `2` | 动量符号 |
| `Nsample` | `3000` | resample 样本数 |
| `dt_max` | `20` | 最大 tsep |
| `ave_dirs` | 6 个方向 | 画平均图时使用的方向列表 |

### Debug 模式

在代码开头修改：

```python
debug = True   # 使用前 5 个组态，输出到 0_debug 目录
jack = False   # 使用普通 bootstrap
```

## 处理流程

```text
OPE 数据 (3 个 .npz / conf)
    ↓ load_ope()
组合为 _ope (Nconf, Nt, Nx)
    ↓
2pt 数据 (1 个 .npy / conf)
    ↓ load_2pt()
加载为 _corr (Nconf, Nt, Nt)
    ↓
compute_ratio()
1. 构建相对坐标 _corr2_rel, _ope_rel, _corr3
2. 对 ti 求平均
3. resample (jackknife / bootstrap)
4. 计算 ratio = (C3 - C2 * OPE) / C2
    ↓
保存 ratio.npy (6 个方向)
    ↓
画图 (6 个方向 + ave 平均)
```

## 注意事项

- 计算量较大（~800 个组态 × 3 个轴 × 2 个方向），建议提交 SLURM 作业运行。
- `dt_list` 会根据 `dt_max` 自动截断，确保不超过 `dt_max - 1`。
- ave 平均图仅用于查看，不保存 `ave/ratio.npy`。
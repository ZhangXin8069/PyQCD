# pyqcd/testing/simple —— qcd-einsum 单功能对照测试

按 **PyQCU `pyqcu/testing/simple`** 的风格组织：每个 `check_*.py` 是可独立运行的脚本，
逐项打印 `PASS/FAIL/SKIP`，任一 `FAIL` 时以非零码退出。

## 设计约定

| 约定 | 说明 |
|---|---|
| 被测对象 | `pyqcd.einsum`（下称 qcd-einsum） |
| 接口对照 | `opt_einsum`（同一函数、同一入参下的行为是否一致） |
| **正确性参考** | **`torch.einsum`**（complex128、CPU），所有数值结果都必须与 torch 一致 |
| 输入数据 | **全部随机生成**（固定 seed 可复现），不依赖任何外部数据文件 |
| 误差口径 | 范数相对误差 `max|Δ| / max|ref|`，契约阈值 `1e-12` |

> `torch.einsum` 覆盖不到的部分（规划辅助函数、BLAS 判定、解析器、后端派发）以
> `opt_einsum` 为参照，并在文件头显式说明；这类文件的数值正确性仍由
> `check_contract*.py` 的 torch 对照兜住。

## 文件一览

| 文件 | 对照的函数 | 参考实现 |
|---|---|---|
| `check_contract.py` | `contract`（含隐式输出、`out=`、一元转置/trace） | **torch.einsum** |
| `check_contract_path.py` | `contract_path`（path / `opt_cost` / `naive_cost` / `PathInfo` 字段 / `shapes=True`） | opt_einsum + **torch 校验计划可执行性** |
| `check_contract_expression.py` | `contract_expression`（首次求值、换输入复用、`out=`、形状不符 fail-closed） | **torch.einsum** |
| `check_shared_intermediates.py` | `shared_intermediates` / `count_cached_ops` | opt_einsum + **torch 校验数值** |
| `check_helpers.py` | `compute_size_by_dict` / `flop_count` / `find_contraction` | opt_einsum |
| `check_paths.py` | `register_path_fn` / `get_path_fn` / 各规划器代价 | opt_einsum |
| `check_blas.py` | `can_blas`（GEMM/DOT/TDOT/OUTER/不可用） | opt_einsum |
| `check_parser.py` | `get_symbol` / `parse_einsum_input` | opt_einsum + **torch 校验解析语义** |
| `check_backends.py` | `get_func` / `has_einsum` / `has_tensordot` / `infer_backend` | opt_einsum |
| `check_specialize.py` | 形态专用路径（注册表、开/关、env 开关、形状回退） | **torch.einsum** |

## 运行

```bash
cd /home/zhangxin/PyQCD

# 一次性跑全部（任一失败 → 非零退出）
bash pyqcd/testing/simple/run-all.sh

# 单个函数
python -m pyqcd.testing.simple.check_contract
python -m pyqcd.testing.simple.check_contract_path --cases 30
python pyqcd/testing/simple/check_parser.py        # 也可直接当脚本运行
```

## 与 `pyqcd/testing/einsum/simple` 的分工

| 目录 | 定位 |
|---|---|
| `pyqcd/testing/simple/`（本目录） | **逐函数 API 对照**：qcd-einsum 与 opt_einsum 在同一函数上的行为一致性，数值以 **torch** 为准 |
| `pyqcd/testing/einsum/simple/` | **子系统级验证**：原生后端/方程正确性、性能基准、专用 kernel 验收、生产工作流时间占比 |

两者互补：本目录回答"接口对不对"，`testing/einsum/simple` 回答"实现快不快、原生路径稳不稳"。

## 已知前置条件

- `torch` 必须是可导入的（本套件的正确性基准）；缺失时相关文件会直接报错而不是跳过，
  以避免"假通过"。
- `check_contract_path.py` 中的 path 手工执行器只用于验证计划本身；它按 numpy
  einsum 的规则逐步收缩，最后一步强制采用请求的输出顺序。

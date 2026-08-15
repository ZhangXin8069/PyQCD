# generate_einsum Tool API 设计对比

## 概述

`generate_einsum` 是 QCD_Master 中 executor（LLM）调用 codegen 的统一入口。按选择类型分为三种：`meson_2pt`、`baryon_2pt`、`multi_hadron_2pt`。

---

## 1. meson_2pt — 单介子 2pt

### 当前状态（待修复）
```python
# skill_utils.py 中硬编码查表
meson = tool_args["meson"]          # "pion"
gamma = tool_args["gamma_snk"]       # "gamma5"
quark, antiquark = _MESON_FLAVORS[meson]  # → ("u", "d")
```

### 目标
直接接受 quark 层的参数，不经过名字查表。

### API
```python
generate_einsum(
    type="meson_2pt",
    quark="u",
    antiquark="d",
    gamma="g5"          # 可选: g1, g2, g3, g4, g5, gt
)
```

### 内部流程
```
meson_operator(quark, antiquark, gamma)  ← 构造 sink
meson_operator(quark, antiquark, gamma)  ← 构造 source
wick_contract_2pt(snk_meson, src_meson)   ← Wick 收缩
pyquda_format_contract(term) for each     ← 生成代码
```

### 拓扑数
| 介子 | 拓扑 |
|---|---|
| 任意 (g5/gt/g1...) | 1 |

---

## 2. baryon_2pt — 单重子 2pt

### 当前状态（待修复）
```python
baryon = tool_args["baryon"]     # "proton"
flavors = {"proton": ("u","d","u"), ...}[baryon]
```

### 目标
直接接受 flavors + projector。

### API
```python
generate_einsum(
    type="baryon_2pt",
    flavors=["u", "d", "u"],     # q1, q2, q3
    projector="P_plus"           # P_plus | P_minus
)
```

### 内部流程
```
baryon_operator(q1, q2, q3)              ← 构造 sink
baryon_operator(q1, q2, q3)              ← 构造 source
wick_contract_2pt(snk_baryon, src_baryon) ← Wick 收缩
pyquda_format_contract(term) for each     ← 生成代码
```

### 拓扑数
| 重子 | 拓扑 |
|---|---|
| 任意 (P_plus) | 2 |
| 任意 (P_minus) | 2 |

### ⚠️ 注意
投影算符 `P_plus` / `P_minus` 由 Tmat 块实现。单重子只需要一个 Tmat 定义。

---

## 3. multi_hadron_2pt — 多强子 2pt

### 适用场景
- 多个介子（如 ππ, DD, KK）
- 多个重子（如 pn, pnΛ）
- 介子+重子混合（如 pπ）
- 反重子（Λ̄）

### API
```python
generate_einsum(
    type="multi_hadron_2pt",
    specs=[
        # 每个 spec = meson_2pt 或 baryon_2pt 的核心输入
        {"type": "meson",   "flavors": ["u", "d"],  "gamma": "g5"},      # π
        {"type": "baryon",  "flavors": ["u", "d", "u"], "projector": "P_plus"},   # p
        {"type": "baryon",  "flavors": ["d", "u", "d"], "projector": "P_plus"},   # n
        {"type": "antibaryon", "flavors": ["u", "d", "s"], "projector": "P_plus"}, # Λ̄
    ]
)
```

### API 设计原则
- **specs 列表的每个元素，就是 meson/baryon 核心参数的直接组合**
- meson spec：`{type, flavors, gamma}` — 等价于 meson_2pt 去掉 `type=` 前缀
- baryon spec：`{type, flavors, projector}` — 等价于 baryon_2pt 去掉 `type=` 前缀
- antibiotics：`{type="antibaryon", flavors, projector}` — 重子 + O.adjoint()
- **每出现一个 baryon/antibaryon，specs 就需要一个 projector**（Tmat_P 或 Tmat_M）

### 内部流程
```
codegen_multi_hadron.gen_code_2pt(specs, specs)
  └─ for spec in specs:
       ├─ type=="meson"       → meson_operator(flavors[0], flavors[1], gamma)
       ├─ type=="baryon"      → baryon_operator(flavors...)
       └─ type=="antibaryon"  → baryon_operator(flavors...).adjoint()
  └─ build_spin_connector(specs)   ← per-baryon 投影独立
  └─ wicklib 全配对
  └─ pyquda_format (自带 codegen)
```

### 拓扑数
| 场景 | 拓扑 |
|---|---|
| 单介子 | 1 |
| 单重子 | 2 |
| 双介子 | 4 |
| 双重子 (p+n) | 36 |
| 重子+介子 (p+π) | 12 |
| 三重子 (p+n+Λ) | 576 |

---

## 4. 三种 API 对比

| 维度 | meson_2pt | baryon_2pt | multi_hadron_2pt |
|---|---|---|---|
| **适用场景** | 单介子 | 单重子 | 任意组合 |
| **Wick 配对** | 1 个配对 | 2 个配对 | 自动枚举 N! |
| **meson_2pt 能覆盖吗** | ✅ 本身 | ❌ | ✅ 包含单介子 |
| **baryon_2pt 能覆盖吗** | ❌ | ✅ 本身 | ✅ 包含单重子 |
| **projector 数量** | 0 | 1 | 每重子一个 |
| **LLM 输入复杂度** | 低 | 低 | 中 |
| **底层复用** | contract.py + codegen.py | 同上 | codegen_multi_hadron.py |

---

## 5. 输出格式差异

```
meson_2pt / baryon_2pt:     仅 sink block（contract() 行），定义/trace/gather 由 Executor 补
multi_hadron_2pt:           完整代码（定义 + sink block + trace + gather + save）
```

**确认**: `gen_code_2pt()` 保持输出完整自包含代码，Executor 直接粘贴即可。

---

## 6. 接入策略

```
skill_utils.py _handle_generate_einsum()
├─ etype == "meson_2pt"
│   ├─ 参数: quark, antiquark, gamma
│   └─ → meson_operator() → wick_contract_2pt() → codegen
├─ etype == "baryon_2pt"
│   ├─ 参数: flavors, projector
│   └─ → baryon_operator() → wick_contract_2pt() → codegen
└─ etype == "multi_hadron_2pt"
    ├─ 参数: specs 列表
    └─ → codegen_multi_hadron.gen_code_2pt(specs, specs)
```

- standard meson_2pt / baryon_2pt → 不走 multi_hadron
- 只有 user 要求多强子或 LLM 无法拆分时才走 multi_hadron

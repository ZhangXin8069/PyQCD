# PEC 结构适配中间层

将用户输入的 einsum 转换成 `pyqcd.einsum.contract`（PEC）已经注册的特殊优化方程，并还原用户要求的输出顺序。2026-11-13 重写版本的设计、接口和验证见 [适配文档](docs/FAST_PATH_ADAPTATION.md)。

```python
from pyqcd_einsum_fork import StructuralEinsum

adapter = StructuralEinsum()
user = "eafb,cqdr,mknl,sitj,ogph,ac,gi,mo,qs,bd,fnp,hjl,rt->ek"
result = adapter.contract(user, *operands, optimize="auto")
info = adapter.explain(user, *operands)
print(info["pec_equation"], info["slot_to_user"], info["execution_mode"])
print(adapter.cache_info())
```

支持操作数重排、全局指标改名、输入轴转置、输出轴恢复，以及完整拓扑的严格匹配。只重排操作数时不会复制张量；改名加操作数重排也会优先寻找无轴转置的映射。匹配成功后的核心计算始终由 PEC 执行注册方程。匹配元数据不代表原生内核一定成功，设备、dtype 和底层安装仍可能导致 PEC 内部回退。

默认缓存保留到显式清空或刷新，也可通过 `cache_size=128` 等设置启用有界 LRU。同一方程的后续调用在解析、注册表搜索和图匹配之前命中缓存；形状、dtype、步长或设备改变时重新建立执行计划。缓存不保存数组内容或结果。未匹配也会缓存。

对于一个需要改写的完整拓扑，第一次真实计算会先各预热一次，再分别计时 PEC 的 `specialize=True` 和 `specialize=False`，把较快的选择写入该方程、形状、布局、后端和 PEC 选项对应的执行缓存；后续调用只执行缓存的 `specialize` 选择。第一次调用因此包含四次 PEC 执行的冷启动成本。这个自适应选择保证中间层不会固定使用较慢的 PEC 分支，但它只比较 PEC 两种模式，不能单独保证一定快于 `opt_einsum`。

```python
from pyqcd_einsum_fork import contract, einsum, explain, contract_many
from pyqcd_einsum_fork import cache_info, clear_cache, refresh, registry_info

values = contract_many([(user, operands), (user, new_operands)])
# 运行中修改 PEC 注册表后刷新；升级安装文件后重启 Python。
refresh()
```

额外大写指标不再触发核心路径的保留、切片或循环处理。只有用户完整方程与某个注册方程拓扑相同（包括这些指标）时才匹配；否则直接回退到 PEC 原式。中间层只做操作数排序、指标改名、轴转置和输出轴恢复。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

# Nev=100、M=9/27；包含三方计时、数值验证及计时外原生分发探针。
# CPU：gpu030 上直接运行；先构建与该节点匹配的 AVX2 测试库。
ssh gpu030 'bash -l /public/home/sush/distillation/einsum/benchmarks/rewrite_build_gpu030_cpu.sh'
ssh gpu030 'bash -l /public/home/sush/distillation/einsum/benchmarks/rewrite_cpu_gpu030.sh'
sbatch benchmarks/rewrite_gpu_debug.slurm
```

CPU 直接运行与 GPU Slurm 脚本使用同一份已归档代码快照保证当前验收可重复。gpu030 的 AVX2 库由经 SHA256 校验的同版本原始源码在工作区构建，通过 `PYQCD_EINSUM_KUPL_LIB` 选择；原安装库不变。CPU 构建源码保存在 `benchmark_output/pyqcd-cpu-gpu030-build/`，库构建与验证记录见任务报告。修改实现后应生成新的快照并更新脚本中的路径，再提交新的作业。CPU 结果为 `benchmark_output/pec-rewrite-cpu-gpu030-avx2-20261113-v2.json`（再次运行需给脚本传入新输出路径）；GPU 结果保存在 `benchmark_output/pec-rewrite-gpu-JOBID.json`。历史基准脚本和结果保留，不能与重写版本混算。

## 直接替代 `pyqcd.einsum.contract`

包级 `contract` 与 PEC 0.4.x 使用相同的主要参数，可以直接赋值替换；它会在每次调用内部判断精确注册、同构重排、完整拓扑匹配或普通 PEC 回退：

```python
import pyqcd.einsum
import pyqcd_einsum_fork

pyqcd.einsum.contract = pyqcd_einsum_fork.contract
value = pyqcd.einsum.contract(
    equation, *operands,
    out=None,
    use_blas=True,
    optimize="auto",
    memory_limit=None,
    backend="numpy",
    specialize=None,
)
```

需要可恢复替换时使用 `install()`：

```python
from pyqcd_einsum_fork import install

with install(patch_einsum=True) as adapter:
    value = pyqcd.einsum.contract(equation, *operands, optimize="auto")
# 离开 with 后恢复原始 pyqcd 函数
```

`install()` 返回的 `ContractPatch` 也可以手动调用 `.restore()`。直接赋值方式使用包级默认适配器；需要独立缓存或独立注册表生命周期时，创建 `StructuralEinsum(base=pyqcd.einsum)`。

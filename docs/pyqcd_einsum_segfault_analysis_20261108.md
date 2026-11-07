# pyqcd.einsum 退出期 Segfault：定位、复现与判定（2026-11-08）

> 结论一句话：**在三条指定路径（原子替换 / spawn 父子 / OpenBLAS 线程）上均未复现**（0/20、0/20、
> 0/120）；**但在真实 bench 负载上用"映射期间就地截断改写 `.so`"可 100% 复现同一症状**（6/6：产物
> 全部写盘后退出期 SIGSEGV），而原子 `mv -Tf` churn 下 1/6。判定为**并发重建竞态**，其充分条件是
> "已映射 inode 被就地截断"而非"原子替换"；`make.sh`/`build-kupl.sh` 本身是原子的，故触发者应是
> 绕过 `make.sh` 的直接链接（`g++ -o build/libpyqcdeinsum_kupl.so`）或
> `setup.py::_stage_native_library()` 的 `shutil.copy2`（`O_TRUNC` 覆盖）。**未修改任何代码**
> （含 `pyqcd/einsum/_native.py`），反证实验见 §8：禁用退出期 native 调用后仍 20/20 崩溃。

---

## 0. 逐条判定表

| 假设 | 实验 | 重复 | 崩溃 | 判定 |
|---|---|---|---|---|
| A：映射期间原子替换/删除 → 退出 `munmap` 崩 | §2，5 种动作 | 5×20=100 | **0** | **未复现**：原子 `rename`/`unlink` 保留旧 inode，映射始终有效，`munmap` 不触页 |
| A′：崩溃只可能发生在"迟到触页" | §3，5 种变体 | 80 | 就地截断 **50/50 崩**；不截断 **0/30** | 机制成立：致命成分是**截断已映射 inode** |
| B：spawn 父子同持 `.so`，父先退出 | §4 | 20 | **0/20**（父崩）、**0/20**（子未完成） | **未复现** |
| C：OpenBLAS 线程池 + fork/spawn 析构 | §5，6 种线程设置 | 6×20=120 | **0/120** | **未复现** |
| 真实负载（L24/E5，42 s/次） | §7 | 28 | 静止 **0/8**；原子 churn **2/14**（两次签名不同，未归因）；尾部就地改写 **6/6** | 见下 |

**对原现象（原命令所遇那一次退出期 Segfault）的判定**：
**未复现**（独立、无并发重建、原子替换条件下 8/8 正常退出），
**判定为并发替换竞态**（证据：(1) §7 的 `tail_inplace` 在**真实 bench 负载**上 6/6 复现出
与事故**完全相同**的签名——产物/结果全部落盘之后退出期 SIGSEGV；(2) §3 定位到最后一次触页
是退出期 `applyDestroyQcd` 与 archive 自身 ELF finalizer；(3) §2 证明单纯的原子替换/删除**无害**，
因此该竞态的具体形态是"**映射期间就地截断改写**"——与 `make.sh --clean` 的删除重建、
绕过 `make.sh` 的直接链接、以及 `setup.py::_stage_native_library()` 的 `shutil.copy2`
（`O_TRUNC` 覆盖）三种可能动作相符；(4) 事故时间窗内确有其他代理在重编该 archive）。

---

## 1. 环境与事实基线

| 项 | 值 |
|---|---|
| 机器 | WSL2，内核 6.18.54.1-microsoft-standard-WSL2，i7-13620H（8 逻辑核），15 GB 内存 |
| Python / NumPy | 3.12.4 (conda-forge) / 2.5.4 |
| 本机 `libgomp`、`libstdc++` | 来自 miniforge（`ldd` 见附录） |
| 被测 archive | `cpp/kupl/einsum/libpyqcdeinsum_kupl.so` → 符号链接 → `build/libpyqcdeinsum_kupl.so`，623072 B，md5 `8ab622f243aa86e5726e383ffb482072` |
| 加载路径解析 | `pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so`（打包副本）**当前不存在**，故 `library_path()` 返回仓库内 `cpp/kupl/einsum/libpyqcdeinsum_kupl.so` |
| `/tmp` 文件系统 | tmpfs（实验脚本与被测副本所在处） |

### 1.1 事故命令的"有效形式"（重要）

裸命令在当前环境里**根本不会加载 `.so`**：

| 命令 | 墙钟 | 退出码 | 是否做 native 工作 |
|---|---|---|---|
| `bench_native --profiles L24 --backends cpu --equations E5` | **0.19 s** | 2 | **否**（冻结 oracle 子模块 `refer/git-rep/opt_einsum` 为空 → `RuntimeError`，一行基准都没跑） |
| `… ` + `PYQCD_EINSUM_ORACLE=installed` | **0.28 s** | 2 | **否**（真实 perambulator 缺失，`DATA ERROR` 直接阻断） |
| `… ` + `PYQCD_EINSUM_ORACLE=installed --data synthetic` | **42.5 s** | 2 | 是（`E5/cpu: opt=0.878268s qcd=0.825404s max_rel=0`） |

因此"同一命令第二次执行未复现（正常 exit=2）"这一对照**不具备可比性保证**：若第二次运行时的
oracle/数据状态与第一次不同，它可能在 0.2 s 内退出、从未 `dlopen` 该 `.so`，自然不可能崩。
后文所有真实负载实验都固定使用第三种形式。

### 1.2 事故现场痕迹

`pyqcd/testing/einsum/artifacts/` 下 `bench_20261108054842`、`bench_20261108055148` 两个目录
各有一份**完整**的 L24/E5 CPU 产物（`bench.json`/`bench.md`/`env.json` 均可解析，含同一
`operand_shapes=[[4,4],[24,24,24,3,2,4,4,50],[4,4]]` 行），与"结果与产物已写盘后才崩"的描述一致。

### 1.3 父进程确实持有该 archive 的映射

`pyqcd/testing/einsum/bench_distillation.py:380-386` 的 `_native_availability()` 调用
`_native._kupl_entry()` → `pyqcd/einsum/kupl/einsum.py:load_plan()` → `ctypes.CDLL(...)`。
`--isolation process` 下父进程不执行任何 contract，但它**已经 dlopen 并映射了该 archive**，
退出时由 glibc 的 `_dl_fini` / `__cxa_finalize` 触发该 archive 自身的 ELF finalizer。
这条事实是 §7 中"静默崩溃"的成因。

---

## 2. 假设 A：`.so` 在被映射期间被替换 / 删除，退出时 `munmap` 是否崩溃？

**实验设计**：子进程用 `ctypes.CDLL(lib, RTLD_LOCAL)` 映射 archive 并成功调用一次
`applyPlanQcd`，随后写 marker 文件；父进程等到 marker（确保映射已存在）后对文件做指定动作，
子进程再调用一次（`__del__` 之外的最后一次触页）后正常退出。逐次记录退出码/信号。

**脚本**：`segv/scripts/runA.py`（驱动）+ `a_child.py` / `a_stress.py`（子进程）

**命令**（每个模式 20 次）：

```bash
python scripts/runA.py <mode> 20      # mode ∈ atomic1|atomic20|unlink|symlink|stress|truncate
```

| mode | 动作 | 重复 | 崩溃 | 信号 |
|---|---|---|---|---|
| `atomic1` | 映射存活时做 **1 次** `rename(2)`（等价 `mv -Tf`）替换 | 20 | **0** | — |
| `atomic20` | 1 秒内 20 次原子替换 | 20 | **0** | — |
| `unlink` | 直接 `unlink` 被映射的文件 | 20 | **0** | — |
| `symlink` | 通过符号链接 dlopen，原子换掉链接目标（复刻 `make.sh:221-222`） | 20 | **0** | — |
| `stress` | 原生调用死循环 1 s，期间父进程做 **163–170 次/次**原子替换 | 20 | **0** | — |
| `truncate`（正对照） | **就地** `truncate(0)` + 写回同样字节 | 20 | **20** | SIGSEGV |

**`/proc/<pid>/maps` 证据**（`atomic1` 首次运行采样，节选）：

```
7debc1665000-7debc16f3000 r-xp 00002000 00:3d 91837  .../work/libA_atomic1.so (deleted)
```

映射仍指向**旧 inode**（`(deleted)`），页表有效，`munmap` 不触页，故不会崩。
**结论：假设 A 不成立（0/100）。** 原子替换/删除对已映射进程是无害的。

---

## 3. 假设 A′：退出期 native 调用是唯一"迟到的触页"

原子替换不产生崩溃，说明必须让**映射本身的页失效**才可能崩。于是把"最后一次触页"精确限定为
退出期的 `applyDestroyQcd`（`pyqcd/einsum/kupl/einsum.py:393-401` 的 `KuplProgram.__del__`）。

**脚本**：`segv/scripts/runA6.py` + `a6_exit.py`
**设计**：子进程通过公开 API `pyqcd.einsum.contract("ab,bc->ac", …)` 编译出一个带 native context
的程序（`live_contexts=1`），写 marker，**等父进程改完文件后**才退出；期间不再做任何 `pyqcd`
调用，因此退出期 destroy 是改写后对该 archive 的第一次访问。

**命令**：`python scripts/runA6.py <action> <mode> <reps>`

| action | mode | 动作 | 重复 | 崩溃 | 崩溃点 |
|---|---|---|---|---|---|
| `control` | `destroy` | 不改文件 | 20 | **0** | — |
| `rewrite` | `destroy` | 退出前就地 truncate+写回 | 20 | **20** | `kupl/einsum.py:398 __del__`（faulthandler 有栈） |
| `rewrite` | `nodestroy` | 改写前已 `applyDestroyQcd` 并把 `context/destroy_function` 置空 | 20 | **20** | **无 faulthandler 栈**（Python 终结之后） |
| `truncate_only` | `destroy` | 只 `truncate(0)`，不回写 | 10 | **10** | SIGBUS(7) |
| `write_keep_size` | `destroy` | 同长度就地覆写、**不 truncate** | 10 | **0** | — |

faulthandler 取证（`rewrite/destroy`，20/20 相同）：

```
Fatal Python error: Segmentation fault
Current thread 0x0000701abfb3d740 (most recent call first):
  Garbage-collecting
  File "/home/zhangxin/PyQCD/pyqcd/einsum/kupl/einsum.py", line 398 in __del__
```

**结论**：
1. 退出期的 `applyDestroyQcd` 确实是"迟到触页"点，且它的失效会导致**产物写完后才崩**；
2. 致命成分是**截断（truncate / O_TRUNC）**，不是"写文件"本身（同长度覆写 0/10）；
3. **关掉这次调用并不能救**（`nodestroy` 仍 20/20），因为 20/20 的崩溃转移到了
   Python 终结之后、由 loader 执行的 archive 自身 finalizer 上（faulthandler 已卸载 → 静默）。
   这一点是 §8 "不改代码"的核心依据。

---

## 4. 假设 B：spawn 父子同时持有同一 `.so`、父进程先退出

**脚本**：`segv/scripts/runB.py` + `b_parent.py` + `b_child.py`
**设计**：父进程 dlopen 并调用一次，`subprocess.Popen` 起一个**全新解释器**子进程（同机、同
archive，各自独立映射），0.4 s 后父进程**不等待子进程**直接正常退出；子进程继续存活 1.0 s、
再调用一次后写 OK marker 退出。逐次记录父退出码与子是否完成。

**命令**：`python scripts/runB.py 20`

**结果**：20 次重复中 —— 父进程崩溃 **0/20**（全部 `parent_rc=0`），子进程未完成 **0/20**。

**结论：假设 B 不成立。** 多进程各自独立映射同一 archive，父先退出没有影响。

---

## 5. 假设 C：OpenBLAS 线程池在 fork / spawn 之后的析构

**脚本**：`segv/scripts/runC.py` + `c_fork.py`
**设计**：`numpy` 1200×1200 `@` 先预热 OpenBLAS 线程池，再 `os.fork()`；子进程继续做 BLAS 并写
OK marker；父进程分 `wait`（等子）与 `parent_first`（先退）两种。`OPENBLAS_NUM_THREADS` 取
1 / 4 / 未设置（=8 核）三种对照。

**命令**：`python scripts/runC.py 20`

| 设置 | 父进程崩溃 | 子进程未完成 |
|---|---|---|
| threads=1, wait | 0/20 | 0/20 |
| threads=1, parent_first | 0/20 | 0/20 |
| threads=4, wait | 0/20 | 0/20 |
| threads=4, parent_first | 0/20 | 0/20 |
| 默认线程, wait | 0/20 | 0/20 |
| 默认线程, parent_first | 0/20 | 0/20 |
| **合计** | **0/120** | **0/120** |

**结论：假设 C 不成立。** `OPENBLAS_NUM_THREADS` 变化对退出期无影响。

---

## 6. 任务第 4 项：`_native.py` 与 kupl 加载/析构逻辑审计

### 6.1 事实（读码）

* `pyqcd/einsum/_native.py` **没有** `atexit`、`__del__`、`weakref`、`dlclose`；它只在被调用时
  `CDLL` 一次并把句柄缓存进模块全局 `_LIBRARY`（`_load_library`，`_LIBRARY_LOCK` 保护），
  因此**`.so` 在整个进程生命周期内始终处于映射状态**，不存在"模块对象已卸载后仍被使用"。
* `load_plan()` / `load_contract()` 返回的 `ctypes._FuncPtr` 自带对 `CDLL` 的引用，不会悬空；
  CPython 的 `ctypes.CDLL` 从不 `dlclose`，所以也不存在"库被卸掉、函数指针还在"的组合。
* 唯一的**退出期 native 调用**来自 `pyqcd/einsum/kupl/einsum.py:393-401`：
  `KuplProgram.__del__` → `self.destroy_function(int(self.context[0]))`（`applyDestroyQcd`）。
  该文件不在本次允许改动范围内。
* **实测时序**（`segv/scripts/e_prog_trace.py`，把 destroy 包一层写 fd 3）：三段脚本（正常退出、
  提前清缓存、强行摘掉 `ctypes` 模块）输出顺序一致：

  ```
  main-returns → (cache-cleared)? → end-of-module → atexit-entered → destroy-called
  ```

  即 `applyDestroyQcd` **确实在解释器终结阶段（`atexit` 之后、模块清理期间）** 被调用
  （GC 回收 `_PROGRAM_CACHE`/`_api._LAST_CONTRACT_PROGRAM` 持有的 `KuplProgram`）。
  这解释了为什么 §3 的 `__del__` 会成为"迟到触页"点。
* 单独压这个退出路径：`segv/scripts/e_prog.py`（plain / cleared 各 20 次）→ **0/20 崩溃**。
  可见"在终结期调用 ctypes"本身不会崩，**只有映射失效时才会**。

### 6.2 次要缺陷（与本次崩溃无直接关系，但建议跟进）

1. `_native.py:127` 的 `_LOAD_FAILURE` 是**永久记忆**：一次瞬时 `dlopen` 失败（例如重建窗口
   内符号链接短暂缺失、`make.sh --clean` 的 `rm -rf` 间隙）会让该进程**此后一直**静默回退到
   纯 Python 规划/执行（`available()` 返回 False，`_kupl_entry()` 恒为 None）。并发重建期间
   这会表现为"性能莫名掉到 Python 实现"。建议：不要记忆 `OSError`，或在下次调用时重新探测。
2. `kupl/einsum.py` 的 `_PROGRAM_CACHE` 满 64 条时按插入序逐出并触发 `__del__` 销毁
   context，而 `_api` 侧仍可能持有同一对象并继续用 `context[0]==0` 调用
   `applyCompiledQcd`。native 侧有 `nullptr`/`magic` 双检查（`apply_contract.cpp:1855-1875`），
   实测不会崩，只是静默返回 `QCD_STATUS_INVALID`；上限到达时会出现"莫名退回 Python"。
3. 退出期 native 调用（第 6.1 条）在**映射有效**时完全安全；问题只在"映射失效"这一外因。

---

## 7. 真实负载实验（决定性）

**统一命令**（与事故等价的最强形式，`--outdir` 指向 scratchpad 以免污染仓库）：

```bash
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_LIB=<private copy> PYTHONFAULTHANDLER=1 \
python pyqcd/testing/einsum/bench_native.py \
    --profiles L24 --backends cpu --equations E5 --data synthetic --outdir <dir>
```

驱动：`segv/scripts/bench_loop.py`（三种模式）+ `bench_exit_wrapper.py`（tail_inplace 用）。

| 模式 | 对 `.so` 的动作 | 重复 | SIGSEGV | 其它退出码 | 产物齐全 |
|---|---|---|---|---|---|
| `static` | 私有副本，**全程不动** | 8 | **0** | 8× exit 2 | 8/8 |
| `atomic_churn` | 每 50 ms 做一次 `rename(2)` 原子替换（单次运行 811–2124 次） | 6 | **1**（rep4） | 5× exit 2 | 6/6（含崩溃那次） |
| `atomic_churn`（第二轮，抓 core） | 同上 | 8 | **1**（rep7，**2.5 s 处启动期崩溃**） | 7× exit 2 | 0/1（未跑到写产物） |
| `tail_inplace` | 全流程跑完、**产物已落盘**后，就地 `truncate(0)`+写回，再让进程退出 | 6 | **6** | — | 6/6（含崩溃那次） |

关键细节：

* `tail_inplace` 崩溃的那 6 次，`bench.json`/`bench.md`/`env.json` **全部可解析**，并含完整行
  `E5/cpu: opt=… qcd=… max_rel=0.000e+00 path=eda94defbc2a` —— 与事故描述"所有结果与产物已写盘"
  逐字吻合；`stderr` 为空（faulthandler 无栈），说明崩在 Python 终结之后的 loader finalizer 上。
* `static` 8 次全部 `exit=2`（`NON_ACCEPTANCE`，与事故的"正常 exit=2"一致），耗时 40.6–46.8 s，
  `E5` 行 qcd 0.75–0.97 s / opt 0.84–1.12 s，`max_rel=0`。
* `atomic_churn` 第一轮 rep4 是"产物齐全 + 空 stderr + SIGSEGV"，耗时 108.9 s（基线 41–47 s）；
  当时机器上有另一个代理的 GPU 全量基准在跑（`ps` 与 `artifacts/bench_20261108060403`、
  `…060610` 可证），内存/CPU 争用严重。
* `atomic_churn` 第二轮 rep7 在 **2.5 s**（启动期，`run(...)` 内部）就崩了，并抓到 core
  （`segv/work/cores/core.196812`，1.3 GB）。core 分析（`segv/scripts/coreinfo.py`）：

  ```
  si_signo=11  si_code=-6 (SI_TKILL)  si_addr=0x3e8000300cc   # si_pid==自身 pid 196812, si_uid==1000
  rip=0x7710112a642c  → /usr/lib/x86_64-linux-gnu/libc.so.6 +0xa642c  (pthread_kill+0x11c)
  ```

  即 **SIGSEGV 是进程自己 `raise()`/`tgkill` 给自己的**（不是访存错误，`si_code` 不是
  `SEGV_MAPERR/SEGV_ACCERR`），崩溃栈尾两帧为 `bench_distillation.py:1652 in main` →
  `bench_native.py:66 in <module>`。这与事故的"退出期崩溃"是**不同现象**，**机制未定位**
  （当时另一个代理正在同步编辑 `pyqcd/testing/einsum/bench_distillation.py`（`git status` 显示
  `M`，+45 行：新增 P1/P2 方程与范数误差口径）并在同机跑 GPU 基准，环境不干净）。

**小结**：原子 churn 的合计观测是 **2/14**，且两次签名互不相同、都无法用"页失效"解释；因此
**不作机制结论**，只作为"并发重建/并发编辑期间运行基准不可信"的实证。真正的可解释崩溃只出现在
`tail_inplace`（就地截断改写）：**6/6**。

### 7.1 崩溃点两处（都由"映射失效"触发）

| 崩溃点 | 触发者 | 可观测性 |
|---|---|---|
| `pyqcd/einsum/kupl/einsum.py:398` `KuplProgram.__del__` → `applyDestroyQcd` | Python 终结期 GC | faulthandler 打出 Python 栈 |
| archive 自身 ELF finalizer（`.fini_array` / C++ 静态析构，含 `std::shared_mutex`、`cblas_api` 的 `static`） | glibc `_dl_fini` / `__cxa_finalize` | **完全静默**（faulthandler 已卸载） |

两处都不在 `pyqcd/einsum/_native.py` 内。

---

## 8. 判定，以及为什么没有改代码

**判定**：并发重建竞态成立，但**充分条件是"已映射 inode 被就地截断式改写"**，而不是原子替换：

* 原子替换 / 删除 / 热 churn：0/100（§2）+ 0/8（§7 `static`）；`atomic_churn` 另有 2/14 观测，
  但两次签名互不一致（一次退出期、一次启动期 `SI_TKILL` 自杀式 SIGSEGV）且都无法用页失效解释，
  按 §7 注**不作机制结论**。
* 就地截断改写：20/20（§3 `rewrite`）、20/20（§3 `truncate_only`→SIGBUS）、6/6（§7 `tail_inplace`）。
* 触发者定位：`make.sh:177-222` 是**原子**的（`-o "$BUILT_LIBRARY.tmp.$$"` + `mv -Tf`，
  公开符号链接也是 `ln -s` + `mv -Tf`），`build-kupl.sh` 只是它的封装；因此事故若确由重建引起，
  触发者应是**绕过 `make.sh` 的直接链接**（`g++ … -o build/libpyqcdeinsum_kupl.so`，ld 会截断
  既有输出）或 **`setup.py:42-53 _stage_native_library()` 的 `shutil.copy2(source, destination)`**
  —— 后者对已存在的 `pyqcd/einsum/kupl/libpyqcdeinsum_kupl.so` 是 `O_TRUNC` 就地覆盖。
  若当时的重建全程走 `make.sh`，则该机制不成立，事故退回"未定位的单次事件"，须按 §10 复测确认。

**为什么不动 `_native.py`（反证实验）**：直觉上的最小改动是"退出期不要再调 native 析构"。§3 的
`rewrite/nodestroy` 已经把这条路走完 —— 提前 `applyDestroyQcd` 并把 `context`/`destroy_function`
置空，使退出期**完全不碰** archive：结果仍然 **20/20 SIGSEGV**（崩溃转移到 loader 自己的
finalizer，且静默）。同理，在 `_native.py` 里加 `atexit` 钩子做"确定性析构"也无济于事：真实事故中
就地改写可能发生在任何时刻，`atexit` 钩子照样在改写之后执行。

结论：**根因（就地改写 archive）不在允许改动的文件里，任何 Python 侧改动都无法消除该类崩溃，
故按要求不改代码**，只给构建侧最小修复建议与验收步骤。

---

## 9. 最小修复建议（构建侧，按优先级）

1. **`setup.py::_stage_native_library()` 改为原子落地**（2 行，风险最低）：
   ```python
   temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
   shutil.copy2(source, temporary)
   os.replace(temporary, destination)      # rename(2)：不会截断既有 inode
   ```
2. **禁止绕过 `make.sh` 直接链接到成品路径**：任何直接 `g++/ld -o build/libpyqcdeinsum_kupl.so`
   都会截断已映射 inode。必须 `-o <tmp> && mv -Tf <tmp> <final>`（`make.sh` 已如此，
   建议在 `cpp/kupl/einsum/AGENTS.md` 里写成硬规则）。
3. **验收期间串行化**：基准/验收运行期间不得有 `make.sh`、`build-kupl.sh`、`pip install`、
   `setup.py build*` 并发。运行前后校验 archive 身份未变（§10 步骤 2 的脚本）。
4. （可选）若希望进程对"外部重建"免疫：`ctypes.CDLL` 前把 archive 复制到进程私有路径
   （例如 `$XDG_RUNTIME_DIR/pyqcd-einsum-<pid>/`）再加载。代价是多一份 623 KB 映射与"重建后
   需重启进程"的语义，**非必需**，仅在构建侧无法保证原子性时才考虑。

---

## 10. 最终验收时应执行的确认步骤

**步骤 1 — 确认构建入口都是原子的**（1 分钟）：

```bash
cd /home/zhangxin/PyQCD
grep -n "TEMP_LIBRARY\|mv -Tf" cpp/kupl/einsum/make.sh          # 期望：tmp + mv -Tf
grep -rn -- "-o .*libpyqcdeinsum_kupl.so" cpp/ build-kupl.sh setup.py   # 期望：无直接 -o 成品路径
grep -n "shutil.copy2" setup.py                                  # 若仍存在→按 §9-1 修
```

**步骤 2 — 验收运行的前后守卫**（每次运行都要做，若身份变化则该次结果作废）：

```bash
LIB=cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so
before=$(stat -c '%d:%i:%s:%Y' "$LIB"; md5sum "$LIB")
for i in $(seq 20); do
  PYQCD_EINSUM_ORACLE=installed PYTHONFAULTHANDLER=1 \
  python pyqcd/testing/einsum/bench_native.py \
      --profiles L24 --backends cpu --equations E5 --data synthetic \
      --outdir /tmp/acc_$i > /tmp/acc_$i.log 2>&1
  echo "run=$i rc=$?"
done
after=$(stat -c '%d:%i:%s:%Y' "$LIB"; md5sum "$LIB")
[ "$before" = "$after" ] && echo "ARCHIVE_UNCHANGED_OK" || echo "ARCHIVE_CHANGED_INVALIDATE_RUN"
```

**通过判据**：20/20 均为 `rc=2`（`NON_ACCEPTANCE` 是当前环境的正常码），无 `139/134/135`，
每个 `/tmp/acc_$i/` 都有完整可解析的 `bench.json`+`bench.md`+`env.json`，且 `ARCHIVE_UNCHANGED_OK`。
参考基线：本次 `static` 8/8 通过（40.6–46.8 s/次）。

**步骤 3 — 若崩溃复发，按下面的签名区分两处崩溃点**：

* `stderr` 有 `Fatal Python error: Segmentation fault … kupl/einsum.py … __del__`
  → 映射在退出期已失效（stale mapping），按 §9 处理；
* `stderr` 为空但 `rc=139` 且产物齐全
  → Python 终结之后、loader finalizer 崩溃，同样是 stale mapping；
* `rc=134`(SIGABRT)/`135`(SIGBUS) 或 worker `returncode=-11`
  → 不是本报告的现象，按 worker 报错文本另查。

**步骤 4 — 复发取证清单**（一次性准备）：

```bash
ulimit -c unlimited; echo "$PWD" > /tmp/core_dir      # 让 core 落到当前目录
cat /proc/sys/kernel/core_pattern                     # 期望 "core"
# 运行期间监听 archive 是否被动过：
inotifywait -m -e modify,attrib,move,delete --format '%T %e %f' \
    cpp/kupl/einsum/build/ cpp/kupl/einsum/ 2>/dev/null | ts
```

若要在 20 次内主动验证"重建竞争"，可用 §7 的 `tail_inplace` 复现器
（`scripts/bench_loop.py tail_inplace 1`，6/6 必崩，产物齐全）作为**负向对照**确认复现能力。

---

## 附录 A：实验脚本与日志

脚本（本次会话 scratchpad，`…/scratchpad/segv/scripts/`）：
`runA.py` / `a_child.py` / `a_stress.py`（假设 A）、`runA6.py` / `a6_exit.py`（A′）、
`runB.py` / `b_parent.py` / `b_child.py`（B）、`runC.py` / `c_fork.py`（C）、
`e_prog.py` / `e_prog_trace.py`（退出期审计）、`bench_loop.py` / `bench_exit_wrapper.py`
/ `churn_core.py`（真实负载）、`repeat.py`（通用重复器）。

逐次记录（JSON，含每次的 exit code / signal / stderr 尾）：
`segv/logs/pathA_*.json`、`pathA6_*.json`、`pathB.json`、`pathC.json`、`repeat_E_*.json`、
`bench_static.json`、`bench_atomic_churn.json`、`bench_tail_inplace.json`、
`churn_core.json`；真实负载产物在 `segv/artifacts/<mode>_<rep>/`。

## 附录 B：`ldd` 依赖（出事 archive）

```
libstdc++.so.6 => /home/zhangxin/miniforge3/lib/libstdc++.so.6
libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6
libgomp.so.1 => /home/zhangxin/miniforge3/lib/libgomp.so.1
libgcc_s.so.1 => /home/zhangxin/miniforge3/lib/libgcc_s.so.1
libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6
```

# I/O reference — HDF5 / ASCII / VdV / VVV / environment

## 适用范围

任务涉及 PyQCD 管线产物、旧格式兼容、L.Liu ASCII 关联函数、预计算顶点积或运行环境
快照时读取。本文件只定义数据交换边界，不定义统计估计量。

## 接口表

| 数据 | API | 纪律 |
|---|---|---|
| 管线张量 | `save_tensor_h5(arr, path, dataset)` / `load_tensor_h5(path)` | 保存 dtype、shape 和 dataset 名 |
| 旧产物 | `_load_any` | 读取优先 `.h5`，再回退 `.npy/.npz` |
| ASCII 关联函数 | `write_data_ascii(data, T, L, filename)` / `read_data_ascii(filename)` | `.gz` 自动压缩/读取；离散标签列在 Re/Im 前 |
| V†V | `readin_vdv_all(vdv_dir, nev, nev1, Nt, ...)` | f8 交错复数，Nev 自探测，按 Nev1 截断 |
| VVV | `readin_vvv_all` / `readin_vvv` | 先核对维度、端序与截断参数 |
| 环境 | `pyqcd.tools._env.dump_env` | 每次管线运行保存 git/包/XeLaTeX/GPU/cmdline |

## 验证顺序

检查输出目录、覆盖策略、shape、dtype、端序、元数据和读写 round-trip。basename-only
路径不得依赖隐式目录创建；跨版本读取必须保留原始后缀和回退路径证据。任何转换
（复数到实数、float 精度或轴重排）都在元数据中说明，不能静默丢失信息。

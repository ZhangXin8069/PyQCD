# AGENTS.md — pyqcd/gauge

纯数组规范观测量：Wilson 矩形圈、Polyakov 圈、Clover 场强与拓扑荷。
输入固定为 `(Nt,Nz,Ny,Nx,4,Nc,Nc)`，方向标签为 `0=x,1=y,2=z,3=t`，输出留在当前后端。

## 边界

- 本子包不处理费米子、MPI 或文件 IO。
- 严格区分逐点密度、体积平均和总和；不得用含糊的 `charge`/`density` 互代。
- 新观测量应加入 `pyqcd.testing` 的规范观测量契约，并保持 numpy/cupy/torch 一致。

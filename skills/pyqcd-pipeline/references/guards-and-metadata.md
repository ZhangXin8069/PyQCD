# 数据守卫、进度与元数据 reference

## 输入守卫

运行前同时检查：

1. 组态目录和必需模板是否存在；
2. 文件大小是否非零且与模板/预期类型一致；
3. 数组能否读取，shape、dtype、有限性和轴序是否符合入口契约；
4. 组态 ID 是否唯一，输出目录是否属于当前运行。

`check_files_existence` 的模板占位符可组合多种文件名；缺失、空文件和大小/内容异常
要归入可解释的 `corrupted`/`missing` 清单，而不是让后续步骤产生空结果。

## 进度日志

使用 `pyqcd.pipeline._validate.ProgressLog` 或 `progress_log` 记录带时间戳的阶段、
组态、状态、耗时和 ETA。ETA 是运行规划信息，不是物理结果；中断时保留已完成与失败
任务，重启后从守卫重新判断。

## 环境快照

每个运行目录保存 `pyqcd.tools._env.dump_env` 生成的 `env.json`，至少含 git 状态/版本、
Python 与依赖版本、XeLaTeX、GPU/后端、命令行、时间和数据路径。快照只记录必要的
环境信息，不写入凭据、token 或原始机密配置。

## 产物状态

建议为每个阶段保存 `started/running/completed/failed/skipped` 状态和原因。完成标记只有
在文件写完、能读回并通过 shape/元数据检查后才能落盘；写临时文件后原子替换，避免
断点续跑把半文件当成完整产物。

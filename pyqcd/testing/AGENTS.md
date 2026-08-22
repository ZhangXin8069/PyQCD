# AGENTS.md — pyqcd/testing

集成测试函数（40 项 = 18 物理/链路项 + 22 整合功能项：一、二轮 stout/eigvec+Ω/CG/hB-loader/boot协方差/plateau+CS核/PDF成图/守卫+续跑/方向能量链/helicity/FH窗/L.Liu ASCII + 三轮匹配核修正/sin准PDF/OPE±z+FF+Lorentz表/宇称投影/ZR样本环/boot外推/分组聚合+dis_connect/模板守卫/Wick图+FLOPs诊断/VdV-VVV读取/env快照/比对原语），另含 NaN 感知比对原语 rel_maxdiff/cmp_one，直接定义于 `__init__.py`，由 examples/pyqcd/conftest.py 导入调度。运行：`python examples/pyqcd/conftest.py`。

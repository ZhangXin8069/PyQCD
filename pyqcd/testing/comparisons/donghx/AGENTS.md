# AGENTS.md — pyqcd/testing/comparisons/donghx

PyQCD 对 donghx 参考实现的数据盘点、低层/费米子/HYP-OPE/2pt 对照与独立验证脚本。

- 参考源码只读；桥接层不得成为生产代码依赖。
- 真实数据路径、组态号、动量方向和产物身份必须在 manifest/汇总中可追溯。
- 运行前优先执行对应 `verify_*`/runner 契约；无输入时如实跳过，不生成伪造通过。

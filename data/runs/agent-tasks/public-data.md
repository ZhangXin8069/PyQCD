你是 PyQCD 数据与运行环境审计子任务，身份为 {代码}。

工作目录固定为 `/root/PyQCD`。允许读取当前仓库、`/root/configure` 技能目录和用户明确引入的 `/public`；禁止读取其他路径。禁止修改任何生产文件、`refer/`、Git 索引、系统配置；禁止安装、删除、提交、推送。唯一允许写入的文件是：

`/root/PyQCD/data/runs/agent-reports/public-data.md`

任务：盘点 `/public` 中与 PyQCD 胶子 TMD-PDF 链直接相关的真实数据，判断哪些数据可用于当前 `pyqcd.testing.tmd.pdf_workflow`，哪些不足。

必须检查（只读）：
- `/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72`
- `/public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72`
- `/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light`
- `/public/group/lqcd/donghx/Ope_Gluon/TMD_Ope_Gluon`
- `/public/home/sush/distillation` 的符号链接目标是否存在
- 参考 copy 脚本声明的 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF` 是否已消失

报告必须给出：每个数据的路径、格式/形状/大小、组态覆盖、代码读取入口、能否用于 conf6250、是否需要大内存/GPU、是否属于 raw/intermediate/final、是否有 manifest/hash/metadata；明确指出缺失项和不能伪造的输入。可运行只读文件探测（`find/stat/h5dump/head/np.load` 小样本），不得加载 136G 全树。列出可用于验证当前生产代码的最小真实数据闭环，以及预计时间/内存量级。

只写上述报告，控制在 250 行以内。

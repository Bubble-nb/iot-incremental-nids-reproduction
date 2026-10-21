# 物联网入侵检测：增量学习复现与消融实验

本项目基于原作者公开的 `prep1` 降采样特征，复现并比较部分跨域增量学习方法。项目**不包含原论文的完整复现**，也没有从全部原始流量重新构建全部模型特征。

## 来源与致谢

- 原作者代码仓库：[jmpr0/incremental-nids](https://github.com/jmpr0/incremental-nids)
- 固定代码提交：`842737730852c7c7c0955257c48ca72a22770343`
- 论文 DOI：`10.1016/j.engappai.2025.110143`

原作者的 MIT 许可证和版权说明保存在 `AUTHOR_LICENSE` 中，原作者代码通过脚本按固定版本下载，本仓库不重新发布该代码。新增实验脚本采用 MIT 许可证，见 `LICENSE`。原始实现、兼容性调整、索引修正和研究方法改动的区别见 `NOTICE.md`。

## 环境准备

使用 Python 3.11。实际记录的依赖版本见 `requirements-observed.txt`。请根据硬件选择 PyTorch 官方 CPU 或 CUDA 安装包；即使安装相同版本，也不保证不同硬件得到完全一致的结果。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-observed.txt
.venv\Scripts\python.exe bootstrap_author_code.py
```

初始化脚本会下载固定提交的公开原作者代码、校验源文件哈希，并分别建立兼容版本和修正索引后的研究版本。已有目录会保留。本项目已在空目录中使用现有本地 Python 环境完成下载和全部方法的单轮运行；尚未验证全新依赖环境的完整安装。

## 运行实验

正向迁移（IoT-NID → TON_IoT）和反向迁移（TON_IoT → IoT-NID）的种子 1 命令如下：

```powershell
.venv\Scripts\python.exe -u run_research.py --source iot_nidd --target ton_iot --seed 1 --epochs 200
.venv\Scripts\python.exe save_label_mapping.py --source iot_nidd --target ton_iot --seed 1
.venv\Scripts\python.exe -u run_research.py --source ton_iot --target iot_nidd --seed 1 --epochs 200
.venv\Scripts\python.exe save_label_mapping.py --source ton_iot --target iot_nidd --seed 1
```

若要对应两个方向的三次运行，再分别以种子 2 和 3 重复相应方向的命令。种子会影响类别顺序、训练内部验证集抽样和模型训练随机性；`prep1` 的训练/测试成员保持固定。因此，不同种子**不代表独立的原始测试划分**。

FT 和 FT-Mem 使用原作者训练框架。Replay 将批次改为 32 条目标域样本和 32 条记忆样本，每轮更新次数与 FT-Mem 相同，但样本曝光分布不同。Weighted 使用带平滑的平方根类别权重，并截断在 `[0.2, 5]`。Combined 同时使用两种调整。这些都是已有方法，本项目不宣称提出了普遍适用的新算法。

## 结果与限制

每次运行保存模型检查点、逐样本预测、logits、分类别指标、验证日志和实际记忆规模。模型检查点依据目标验证集上的未加权交叉熵选择。复用检查点前会核对已有实验配置。

组合方法首轮的提升没有在后续运行中稳定重现。报告时应保留全部运行结果，并同时讨论误报率和旧知识保留代价；不得使用测试集调参。固定规则阈值校准由evaluate_fixed_calibration.py执行，覆盖两方向各三次既有检查点；规则起源于首轮结果后的分析，属于事后跨运行核验。

本仓库不包含原始流量、模型权重、逐样本预测、私人凭证、学生个人信息或下载的第三方论文。这里记录的是已完成的本地研究范围；单轮演示不能代表论文结果复现，也不能证明每种依赖安装都已验证。

`manifest.json` 按 GitHub 归档中的 LF 换行记录公开文本文件哈希；代码 ZIP 也统一使用 LF。`source_manifest.json` 则记录原作者源文件未经改动时的字节哈希。

## 目录导航

- `results/README.md`：已完成实验结果索引与评估口径。
- `results/forward_repeats.md`：正向三次运行的全部结果及配对差值。
- `results/reverse_repeats.md`：反向三次结果，配合 `results/repeated_run_metrics.csv` 查看。
- `results/核验结果说明.md`：完全一致输入排除分析，完整口径和逐类支持数。
- `NOTICE.md`：代码来源、修改范围和数据归属说明。


## 输入敏感性核验

完成两方向种子1、2、3的正式运行和标签映射后，执行 `python audit_input_sensitivity.py`。它重建数据加载顺序、核对保存检查点和逐样本logits，再排除两域训练与验证部分已出现的完全相同float32输入。默认输出到仓库父目录的“数据与结果/重复输入敏感性”。未重新训练、未改变分类阈值，不能代替独立网络验证。

正向全量样本中回放三次改善旧域；过滤后对应差值为+4.50、+6.14、−0.33个百分点。原有结论只适用于原测试输入构成，不能称为跨构成一致改善。完整聚合结果与逐类支持数随仓库保留。新增训练与检查点复核的具体环境见 `environment_research_2026-10-05.json`，依赖见 `requirements-research-20261005.txt`；PyTorch CUDA 12.8轮子应按官方安装入口选择，而非由无后缀版本号推定。


## 2026年10月补充脚本

- `evaluate_fixed_calibration.py`：从已保存检查点恢复目标验证输出，在固定规则下复算阈值运行点；不训练。
- `run_author_bic.py`：独立训练作者BiC代码实现；保留作者损失转float行为，不能视为修复后的标准算法。
- `run_full_autodl.sh`：独立输出目录中的正式种子1完整录制入口，最大200轮；`check_recorded_run.py`显示实际分数与论文参照的差异。

这些补充脚本与本地课程提交包同步维护。

## 编程辅助说明

代码准备过程中使用了 AI 编程辅助；实验数值以实际运行日志、保存预测及核验记录为准。原作者代码、项目改动及数据出处分别注明。

## 校准与作者BiC补充对照

`evaluate_fixed_calibration.py`从保存检查点恢复目标验证概率，用0.5%、1%、5%的验证误报预算复算全部六次运行、五个增量方法与双域结果，不重新训练。`results/fixed_calibration`保存180行运行点、冻结规则与输入哈希。该规则源于查看首轮测试后的分析，属于事后跨运行核验。

`run_author_bic.py`调用固定作者代码中的BiC实现。它另行训练源模型和偏置层，与共享源模型的五分支消融分开解读。`results/author_bic`提供两方向种子1的实测记录。

`evaluate_field_permutation.py`按原模型重新计算两方法、两域、四字段、五次置换共80条记录，核对原始logits后才计算指标。运行命令为`python evaluate_field_permutation.py`。正数delta表示置换后Macro-F1下降，不是提升。

`results/review_20261020`包含全部方法逐类指标（含Source-only）、预算敏感性汇总与字段置换记录。完整训练脚本与单轮演示的数值不能混用；未将流程演示标为正式训练录像。

## 实验设计复核（2026-10-20）

新增 `audit_design_20261020.py` 和 `audit_label_conflicts_20261020.py`，结果见 `results/design_audit/`。前者读取六次主实验保存的72份预测与训练日志，复核首轮逐类贡献、稀有类敏感性、回放抽样曝光、类别权重及0.5%验证误报预算的二项区间；后者从固定样例重建输入，统计相同数值特征对应的标签冲突。

72份逐样本预测（真实标签、预测标签和logits；60份增量模型输出17类，12份源模型输出10或12类）随课程提交包 `06数据核验与结果/主实验逐样本预测` 提供，公开仓库暂不托管这些二进制结果。运行第一脚本时用 `--results-root` 指向提交包中的该目录。第二脚本需要先初始化原作者固定样例与预处理数据，并保留对应逐样本预测以检查标签顺序。复核依赖见 `requirements-audit.txt`。新结果是对既有运行的复核，没有新增训练。

## 从归档预测免训练复算校准

`recompute_calibration_from_saved.py` 只用提交包中的72份主实验预测、30份目标验证logits和训练日志，重新选择0.5%、1%、5%阈值，并复算180个双域运行点。它不加载模型或训练数据，不需要GPU。脚本逐项核对原CSV，并核对验证logits的交叉熵是否匹配30个检查点的最佳验证损失。

在课程提交包内，可直接运行 `python 03代码/补充核验/recompute_calibration_from_saved.py`；从公开仓库运行时，以 `--main-results` 指向提交包的“主实验逐样本预测”，以 `--calibration-results` 指向“固定阈值校准”。依赖见 `requirements-audit.txt`。

## 相同输入组的误报分解

`results/test_collision_groups`保存两域最大正常/攻击异标签输入组的构成、72行预测分解和144行逐类正确数。`audit_test_collision_groups.py`核验特征相等性；`recompute_group_attribution.py`只用提交包中的分组索引、预测与验证logits复算648个数值，无需GPU。12份分组NPZ与主实验预测保存在课程提交包的`06数据核验与结果`，不在公开仓库托管。

公开仓库使用复算脚本时，分别通过`--main-results`、`--calibration-results`和`--group-results`指向提交包的主实验逐样本预测、固定阈值校准和测试组误报分解目录。

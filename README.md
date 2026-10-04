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

若要对应已经完成的正向三次运行，再分别以种子 2 和 3 重复正向命令。种子会影响类别顺序、训练内部验证集抽样和模型训练随机性；`prep1` 的训练/测试成员保持固定。因此，不同种子**不代表独立的原始测试划分**。

FT 和 FT-Mem 使用原作者训练框架。Replay 将批次改为 32 条目标域样本和 32 条记忆样本，每轮更新次数与 FT-Mem 相同，但样本曝光分布不同。Weighted 使用带平滑的平方根类别权重，并截断在 `[0.2, 5]`。Combined 同时使用两种调整。这些都是已有方法，本项目不宣称提出了普遍适用的新算法。

## 结果与限制

每次运行保存模型检查点、逐样本预测、logits、分类别指标、验证日志和实际记忆规模。模型检查点依据目标验证集上的未加权交叉熵选择。复用检查点前会核对已有实验配置。

组合方法首轮的提升没有在后续运行中稳定重现。报告时应保留全部运行结果，并同时讨论误报率和旧知识保留代价；不得使用测试集调参。后续阈值探索是在检查首轮结果后设计的，属于探索性分析，不包含在本仓库的最小训练流程中。

本仓库不包含原始流量、模型权重、逐样本预测、私人凭证、学生个人信息或下载的第三方论文。这里记录的是已完成的本地研究范围；单轮演示不能代表论文结果复现，也不能证明每种依赖安装都已验证。

`manifest.json` 按 GitHub 归档中的 LF 换行记录公开文本文件哈希；代码 ZIP 也统一使用 LF。`source_manifest.json` 则记录原作者源文件未经改动时的字节哈希。

## 目录导航

- `results/README.md`：已完成实验结果索引与评估口径。
- `results/forward_repeats.md`：正向三次运行的全部结果及配对差值。
- `results/reverse_single_run.md`：反向单次运行及误报代价。
- `NOTICE.md`：代码来源、修改范围和数据归属说明。

代码准备过程使用了 AI 编程辅助。项目作者仍须负责审查代码、实验结果、学术表述和来源归属。

# 来源、修改范围与限制

## 原作者实现

原作者代码来自 [jmpr0/incremental-nids](https://github.com/jmpr0/incremental-nids)，固定提交为 `842737730852c7c7c0955257c48ca72a22770343`。原始许可证未经修改地保存在 `AUTHOR_LICENSE` 中。

`prepare_smoke.py` 创建独立的兼容性副本；`setup_research.py` 创建重置索引后的研究副本。对应差异记录在两个 `.diff` 文件中。这些是工程兼容和取数修正，不是新的学习算法。

## 本项目的实验改动

`run_research.py` 使用原作者的 CNN 和增量训练框架，增加兼容 GPU 的评估方式，以及预先设定的回放与类别加权消融，并记录固定样例的适用范围。均衡回放和类别加权是已有技术；本项目不宣称首次提出这些方法。

## 数据来源与引用

数据继续遵循各自发布者的使用条款。代码许可证不替代数据授权或引用要求。

- TON_IoT：遵循 [UNSW 官方数据集页面](https://research.unsw.edu.au/projects/toniot-datasets)列出的八项来源引用要求。
- IoT-NID：引用数据集 DOI `10.21227/q70p-q449`。

代码准备过程中使用了 AI 编程辅助。项目作者负责审查代码、结果、学术主张及来源归属。

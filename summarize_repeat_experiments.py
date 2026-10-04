"""Recompute completed run metrics; do not combine migration directions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

root = Path(__file__).resolve().parent
records = []
pending = []
for folder in sorted((root / 'research_results').glob('*_row_index_fixed')):
    protocol_path = folder / 'protocol.json'
    mapping_path = folder / 'label_mapping.json'
    if not protocol_path.exists() or not mapping_path.exists():
        pending.append(folder.name)
        continue
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))['global_labels']
    benign = next(int(k) for k, v in mapping.items() if v == 'benign')
    source, tail = folder.name.split('_to_', 1)
    target = tail.split('_seed', 1)[0]
    for method in ['source', 'ft', 'ft_mem', 'replay', 'weighted', 'combined']:
        if method != 'source' and not (folder / method / 'complete.json').exists():
            pending.append(folder.name + '/' + method)
            continue
        for domain in ['source', 'target']:
            pred_path = folder / method / f'{domain}_predictions.npz'
            if not pred_path.exists():
                continue
            data = np.load(pred_path)
            y, p = data['y_true'], data['y_pred']
            assert np.array_equal(data['logits'].argmax(1), p)
            f1 = f1_score(y, p, labels=np.unique(y), average='macro', zero_division=0)
            reported = json.loads((folder / method / f'{domain}_metrics.json').read_text())
            assert abs(f1 - reported['macro_f1']) < 1e-12
            normal, attacks = y == benign, y != benign
            records.append(dict(direction=source+' -> '+target, seed=protocol['training_seed'],
                method=method, domain=domain, n=len(y), macro_f1=f1,
                accuracy=accuracy_score(y, p), false_positive_rate=float((p[normal] != benign).mean()),
                attack_recall=float((p[attacks] != benign).mean())))
df = pd.DataFrame(records)
df.to_csv(root / 'research_results/repeated_run_metrics.csv', index=False, encoding='utf-8-sig')
lines = ['# 重复与反向实验动态汇总', '',
         '所有运行仍使用作者 prep1 固定训练/测试集合；运行种子还影响类别排序与训练内验证抽样，不能视为三个独立测试划分。',
         '下表只纳入已有标签映射且预测文件独立复算通过的结果。不同方向分别报告，不合并成整体分数。', '',
         '| 方向 | 种子 | 方法 | 域 | Macro-F1/% | FPR/% | 攻击召回/% |',
         '|---|---:|---|---|---:|---:|---:|']
for row in records:
    lines.append(f"| {row['direction']} | {row['seed']} | {row['method']} | {row['domain']} | "
                 f"{100*row['macro_f1']:.2f} | {100*row['false_positive_rate']:.2f} | {100*row['attack_recall']:.2f} |")
lines += ['', '## 尚未完成或待映射的输出', '']
lines.extend('- '+item for item in pending)
(root / '重复与反向实验动态汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(f'Independently recomputed {len(records)} domain results; pending items {len(pending)}.')

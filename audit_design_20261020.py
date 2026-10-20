"""Read-only audit of saved predictions and the fixed experiment protocol."""
from pathlib import Path
import argparse
import csv, json, math
from collections import defaultdict
import numpy as np
from scipy.stats import beta, binomtest, ttest_1samp
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--results-root',type=Path,help='Directory with six seed*_row_index_fixed run folders')
parser.add_argument('--output-root',type=Path,help='Directory for recomputed CSV and JSON')
args=parser.parse_args()
local_results = ROOT / 'research_results'
submission_results = ROOT.parent.parent / '06数据核验与结果' / '主实验逐样本预测'
RESULTS = args.results_root if args.results_root else (local_results if local_results.is_dir() else submission_results)
OUT = (ROOT.parent / '数据与结果' / '实验设计复核') if local_results.is_dir() else (ROOT.parent.parent / '06数据核验与结果' / '实验设计复核')
if args.results_root:
    OUT=args.output_root if args.output_root else ROOT/'results'/'design_audit_recomputed'
OUT.mkdir(parents=True, exist_ok=True)
runs = sorted(p for p in RESULTS.glob('*_seed[123]_row_index_fixed') if p.is_dir())
assert len(runs) == 6, len(runs)

rows = []
rare_rows = []
prediction_files = []
for folder in runs:
    mapping = json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    proto = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    direction = folder.name.split('_seed')[0]
    for domain in ('source', 'target'):
        reference = np.load(folder/'ft_mem'/f'{domain}_predictions.npz')
        y = reference['y_true']
        labels = np.unique(y)
        support = {int(k):int((y == k).sum()) for k in labels}
        regular = [int(k) for k in labels if support[int(k)] > 20]
        for method in ('source','ft','ft_mem','replay','weighted','combined'):
            path = folder/method/f'{domain}_predictions.npz'
            assert path.is_file(), path
            z = np.load(path)
            assert np.array_equal(y,z['y_true'])
            assert np.array_equal(z['y_pred'],z['logits'].argmax(1))
            prediction_files.append(path)
            _,_,f1,sup = precision_recall_fscore_support(y,z['y_pred'],labels=labels,zero_division=0)
            _,_,f1_regular,_ = precision_recall_fscore_support(y,z['y_pred'],labels=regular,zero_division=0)
            rows.append(dict(run=folder.name,direction=direction,domain=domain,method=method,
                n=len(y),classes=len(labels),classes_support_gt20=len(regular),
                macro_f1=100*float(f1.mean()),macro_f1_support_gt20=100*float(f1_regular.mean())))
            if direction=='iot_nidd_to_ton_iot' and folder.name.endswith('seed1_row_index_fixed') and domain=='target' and method in ('ft_mem','combined'):
                for k,f,n in zip(labels,f1,sup):
                    rare_rows.append(dict(method=method,label=mapping[str(int(k))],support=int(n),f1_percent=100*float(f)))

assert len(prediction_files)==72
def write_csv(name, data):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
write_csv('逐运行稀有类敏感性.csv',rows)
by_method={(r['method'],r['label']):r for r in rare_rows}
contrib=[]
for label in sorted({r['label'] for r in rare_rows}):
    base=by_method[('ft_mem',label)]
    combined=by_method[('combined',label)]
    delta=combined['f1_percent']-base['f1_percent']
    contrib.append(dict(label=label,support=base['support'],ft_mem_f1=base['f1_percent'],
        combined_f1=combined['f1_percent'],delta_f1_points=delta,
        contribution_to_macro_f1_points=delta/len(rare_rows)*2))
contrib.sort(key=lambda r:r['contribution_to_macro_f1_points'],reverse=True)
write_csv('正向种子1逐类贡献.csv',contrib)

weights=[]
for folder in runs:
    log=(folder/'combined'/'training.jsonl').read_text(encoding='utf-8').splitlines()
    vals=[r['value'] for line in log if (r:=json.loads(line)).get('group')=='protocol' and r.get('name')=='class_weights']
    assert len(vals)==1
    mapping=json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    for k,v in enumerate(vals[0]):
        weights.append(dict(run=folder.name,label=mapping[str(k)],weight=float(v),at_upper_clip=math.isclose(v,5),at_lower_clip=math.isclose(v,.2)))
write_csv('训练类别权重.csv',weights)

budgets=[]
for folder in runs:
    p=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    mapping=json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    benign=int(next(k for k,v in mapping.items() if v=='benign'))
    n=int(p['validation_counts'][1].get(str(benign),0))
    assert n in (625,637), (folder,n)
    k=math.floor(.005*n)
    # Clopper-Pearson two-sided 95% upper limit at k errors in n Bernoulli trials.
    upper=beta.ppf(.975,k+1,n-k)
    budgets.append(dict(run=folder.name,benign_validation_n=n,max_empirical_false_alarms=k,
        maximum_empirical_fpr_percent=100*k/n,
        exact_two_sided_95pct_upper_fpr_percent=100*upper))
write_csv('验证预算精度.csv',budgets)

gains=[]
for folder in runs:
    if folder.name.startswith('iot_nidd_to_ton_iot'):
        selected={r['method']:r['macro_f1'] for r in rows if r['run']==folder.name and r['domain']=='source'}
        gains.append(selected['replay']-selected['ft_mem'])
summary=dict(prediction_files=len(prediction_files),top_contributions=contrib[:4],
    combined_minus_ft_mem_macro_f1=sum(r['contribution_to_macro_f1_points'] for r in contrib),
    benign_weights=[r for r in weights if r['label']=='benign'],
    clipped_upper_by_run={p.name:sum(r['at_upper_clip'] for r in weights if r['run']==p.name) for p in runs},
    budgets=budgets,
    replay_seed1_expected_old_draws_per_epoch=32*math.ceil((17736+231)/64),
    replay_seed1_old_draws_per_memory_item=32*math.ceil((17736+231)/64)/231,
    replay_seed1_expected_target_unique_fraction=1-(1-1/17736)**(32*math.ceil((17736+231)/64)),
    forward_replay_gain_mean=float(np.mean(gains)),
    forward_replay_gains=gains,
    forward_replay_gain_ttest_two_sided_p=float(ttest_1samp(gains,0).pvalue),
    sign_test_all_positive_two_sided_p={str(n):float(binomtest(n,n,.5).pvalue) for n in (3,5,6)},
    limitation='Rare-class threshold and all statistical checks are retrospective; repeated seeds share fixed test samples. Binomial intervals illustrate ideal independent fixed-threshold resolution and do not correct adaptive model/threshold selection or dependent duplicate inputs.')
(OUT/'复核摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

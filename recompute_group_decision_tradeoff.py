"""Hold other predictions fixed and replace the mixed-label TON_IoT group's decision."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from sklearn.metrics import f1_score

base=Path(__file__).resolve().parent
stage=base.parents[1] if base.name=='补充核验' else base
ev=stage/'06数据核验与结果'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--main-results',type=Path,default=ev/'主实验逐样本预测')
p.add_argument('--group-results',type=Path,default=ev/'测试组误报分解')
p.add_argument('--output-root',type=Path,default=ev/'测试组误报分解')
a=p.parse_args();run='iot_nidd_to_ton_iot_seed1_row_index_fixed';folder=a.main_results/run
names=json.loads((folder/'label_mapping.json').read_text('utf-8'))['global_labels']
benign=next(int(k) for k,v in names.items() if v=='benign')
mitm=next(int(k) for k,v in names.items() if v.lower()=='mitm-arpspoof')
with np.load(a.group_results/'group_membership'/f'{run}_target.npz',allow_pickle=False) as z:indices=z['indices'];group_y=z['y_true']
with np.load(folder/'combined/target_predictions.npz',allow_pickle=False) as z:y=z['y_true'];raw=z['y_pred']
with np.load(folder/'ft_mem/target_predictions.npz',allow_pickle=False) as z:assert np.array_equal(y,z['y_true']);baseline=z['y_pred']
assert np.array_equal(y[indices],group_y)
assert len(indices)==396 and (group_y==benign).sum()==266 and (group_y==mitm).sum()==130
assert np.all(raw[indices]==mitm) and np.all(baseline[indices]==benign)
labels=np.unique(y)
def macro(pred):return 100*float(f1_score(y,pred,labels=labels,average='macro',zero_division=0))
all_mitm=raw.copy();all_mitm[indices]=mitm
all_benign=raw.copy();all_benign[indices]=benign
rest=np.ones(len(y),bool);rest[indices]=False
assert np.array_equal(all_mitm[rest],all_benign[rest])
gap=macro(raw)-macro(baseline);effect=macro(all_mitm)-macro(all_benign)
result=dict(direction='IoT-NID to TON_IoT',seed=1,model='Combined',group_n=396,
    group_benign_n=266,group_mitm_n=130,benign_test_n=int((y==benign).sum()),mitm_test_n=int((y==mitm).sum()),
    ft_mem_macro_f1=macro(baseline),group_all_mitm_macro_f1=macro(all_mitm),group_all_benign_macro_f1=macro(all_benign),
    group_decision_effect_points=effect,combined_ft_mem_gap_points=gap,group_effect_fraction_of_gap=effect/gap,
    interpretation='Exact prediction replacement with all other Combined predictions fixed. This is a reference-dependent arithmetic comparison, not a causal attribution of training effects. It differs from the two-order Shapley calibration decomposition.',
    training_performed=False)
a.output_root.mkdir(parents=True,exist_ok=True)
(a.output_root/'fixed_other_predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
rows=[]
for condition,pred in [('FT-Mem',baseline),('Combined: group all MITM',all_mitm),('Combined: group all benign',all_benign)]:
    values=f1_score(y,pred,labels=labels,average=None,zero_division=0)*100
    for label,f1 in zip(labels,values):rows.append(dict(condition=condition,label=names[str(int(label))],support=int((y==label).sum()),f1_percent=float(f1)))
with (a.output_root/'fixed_other_predictions_per_class.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,ensure_ascii=False))

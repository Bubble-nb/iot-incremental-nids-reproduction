"""Recompute collision-group attribution from archived masks and logits, without training/data/GPU."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from sklearn.metrics import f1_score

def softmax(z):
    z=np.asarray(z,dtype=np.float64);e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def macro(y,p):return 100*float(f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0))

root=Path(__file__).resolve().parent
stage=root.parents[1] if root.name=='补充核验' else root
evidence=stage/'06数据核验与结果'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--main-results',type=Path,default=evidence/'主实验逐样本预测')
p.add_argument('--calibration-results',type=Path,default=evidence/'固定阈值校准')
p.add_argument('--group-results',type=Path,default=evidence/'测试组误报分解')
p.add_argument('--output',type=Path,default=evidence/'测试组误报分解'/'独立复算.json')
a=p.parse_args()
with (a.group_results/'group_prediction_attribution.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
checks=0;cal_rows=0
for row in rows:
    folder=a.main_results/row['run'];domain=row['domain'];method=row['method']
    mapping=json.loads((folder/'label_mapping.json').read_text('utf-8'))['global_labels']
    benign=next(int(k) for k,v in mapping.items() if v=='benign')
    with np.load(folder/method/f'{domain}_predictions.npz',allow_pickle=False) as z:
        y=z['y_true'];raw=z['y_pred'];logits=z['logits']
    assert np.array_equal(raw,logits.argmax(1))
    with np.load(a.group_results/'group_membership'/f'{row["run"]}_{domain}.npz',allow_pickle=False) as g:
        indices=g['indices'];assert np.array_equal(y[indices],g['y_true'])
        assert len(y)==int(g['test_n']) and str(g['input_sha256'])==row['group_sha256']
    group=np.zeros(len(y),bool);group[indices]=True;normal=y==benign
    def check(key,value):
        global checks
        assert np.isclose(float(row[key]),value,rtol=0,atol=1e-10),(row['run'],domain,method,key,row[key],value)
        checks+=1
    check('raw_fp',((raw!=benign)&normal).sum())
    check('raw_group_fp',((raw!=benign)&normal&group).sum())
    check('raw_group_fpr_points',100*((raw!=benign)&normal&group).sum()/normal.sum())
    check('raw_macro_f1',macro(y,raw))
    if method=='source':continue
    val=a.calibration_results/f'{row["direction"]}_seed{row["seed"]}'/method/'validation_predictions.npz'
    with np.load(val,allow_pickle=False) as v:pb=softmax(v['logits'])[v['y_true']==benign,benign]
    values,counts=np.unique(pb,return_counts=True);below=np.r_[0,np.cumsum(counts)[:-1]]
    tau=float(values[below<=int(np.floor(.005*len(pb)))][-1]);check('threshold',tau)
    pr=softmax(logits);non=pr.copy();non[:,benign]=-1;pred=non.argmax(1);pred[pr[:,benign]>=tau]=benign
    only_group=raw.copy();only_group[group]=pred[group]
    only_other=pred.copy();only_other[group]=raw[group]
    delta=.5*(macro(y,only_group)-macro(y,raw)+macro(y,pred)-macro(y,only_other))
    check('cal_fp',((pred!=benign)&normal).sum())
    check('cal_group_fp',((pred!=benign)&normal&group).sum())
    check('cal_macro_f1',macro(y,pred))
    check('group_shapley_macro_delta',delta)
    check('other_shapley_macro_delta',macro(y,pred)-macro(y,raw)-delta)
    cal_rows+=1
assert len(rows)==72 and cal_rows==60
report=dict(status='PASS',method_domain_rows=len(rows),calibrated_rows=cal_rows,numeric_checks=checks,
            training_performed=False,gpu_required=False,
            scope='Checks archived group memberships and prediction attribution; original feature equality is checked separately by audit_test_collision_groups.py.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(report,ensure_ascii=False))

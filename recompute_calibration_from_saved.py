"""Recompute all 180 calibration operating points from archived NPZ files only.

Requires NumPy and scikit-learn; does not load models, training data or CUDA.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
submission=ROOT.parent.parent/'06数据核验与结果'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--main-results',type=Path,default=ROOT/'research_results' if (ROOT/'research_results').is_dir() else submission/'主实验逐样本预测')
p.add_argument('--calibration-results',type=Path,default=ROOT/'research_results'/'fixed_calibration_20261019' if (ROOT/'research_results').is_dir() else submission/'固定阈值校准')
p.add_argument('--output-root',type=Path,default=submission/'校准独立复算' if submission.is_dir() else ROOT/'results'/'saved_calibration_recomputed')
a=p.parse_args()
assert a.main_results.is_dir() and a.calibration_results.is_dir(), 'Supply --main-results and --calibration-results paths from the submission package.'
a.output_root.mkdir(parents=True,exist_ok=True)
hashes={}

def record_hash(path):
    hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()

def softmax(logits):
    z=np.asarray(logits,dtype=np.float64)
    assert z.ndim==2 and np.isfinite(z).all()
    exp=np.exp(z-z.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)

def metrics(y,pred,b):
    normal=y==b;attack=~normal
    assert normal.any() and attack.any()
    return dict(macro_f1=float(f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0)),
        fpr=float(np.mean(pred[normal]!=b)),attack_recall=float(np.mean(pred[attack]!=b)),
        n=len(y),n_benign=int(normal.sum()))

def key(row):
    return (row['source'],row['target'],int(row['seed']),row['method'],row['domain'],float(row['budget']))

expected_path=a.calibration_results/'operating_points.csv'
record_hash(expected_path)
with expected_path.open(encoding='utf-8-sig',newline='') as handle:
    expected_rows=list(csv.DictReader(handle))
expected={key(r):r for r in expected_rows}
assert len(expected)==len(expected_rows)==180
rows=[];checks=[]
for source,target in [('iot_nidd','ton_iot'),('ton_iot','iot_nidd')]:
    for seed in (1,2,3):
        folder=a.main_results/f'{source}_to_{target}_seed{seed}_row_index_fixed'
        mapping_file=folder/'label_mapping.json';record_hash(mapping_file)
        mapping=json.loads(mapping_file.read_text(encoding='utf-8'))['global_labels']
        b=next(int(k) for k,v in mapping.items() if v=='benign')
        for method in ('ft','ft_mem','replay','weighted','combined'):
            validation_path=a.calibration_results/f'{source}_to_{target}_seed{seed}'/method/'validation_predictions.npz'
            record_hash(validation_path)
            with np.load(validation_path,allow_pickle=False) as z:
                yv=z['y_true'];pv=softmax(z['logits'])
            normal_prob=pv[yv==b,b]
            values,counts=np.unique(normal_prob,return_counts=True)
            below=np.r_[0,np.cumsum(counts)[:-1]]
            thresholds={budget:float(values[below<=int(np.floor(budget*len(normal_prob)))][-1]) for budget in (.005,.01,.05)}
            log_path=folder/method/'training.jsonl'
            record_hash(log_path)
            loss_records=[r['value'] for line in log_path.read_text(encoding='utf-8').splitlines()
                if (r:=json.loads(line)).get('group')=='valid' and r.get('name')=='loss']
            val_ce=float(-np.log(pv[np.arange(len(yv)),yv]).mean())
            assert abs(val_ce-min(loss_records))<1e-5,(source,target,seed,method,'validation checkpoint loss')
            checks.append(dict(source=source,target=target,seed=seed,method=method,
                validation_logits_match_best_logged_loss=True,validation_ce=val_ce))
            for domain in ('source','target'):
                path=folder/method/f'{domain}_predictions.npz';record_hash(path)
                with np.load(path,allow_pickle=False) as z:
                    y=z['y_true'];logits=z['logits'];raw=z['y_pred']
                assert np.array_equal(raw,logits.argmax(1)),path
                probabilities=softmax(logits)
                base=metrics(y,raw,b)
                for budget,threshold in thresholds.items():
                    pa=probabilities.copy();pa[:,b]=-1
                    pred=pa.argmax(1);pred[probabilities[:,b]>=threshold]=b
                    result=metrics(y,pred,b)
                    row=dict(source=source,target=target,seed=seed,method=method,domain=domain,budget=budget,
                        benign_label=b,threshold=threshold,val_benign_n=len(normal_prob),
                        val_fpr=float((normal_prob<threshold).mean()),**result,
                        **{'baseline_'+k:v for k,v in base.items() if k in ('macro_f1','fpr','attack_recall')})
                    reference=expected[key(row)]
                    for field,value in row.items():
                        if isinstance(value,(int,float)):
                            assert np.isclose(value,float(reference[field]),rtol=0,atol=1e-12),(key(row),field,value,reference[field])
                        else:assert value==reference[field]
                    rows.append(row)
assert len(rows)==180
with (a.output_root/'operating_points_recomputed.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(a.output_root/'verification.json').write_text(json.dumps(dict(operating_points=180,
    all_archived_values_match_atol=1e-12,checkpoint_validation_loss_checks=checks,
    training_performed=False,model_loading_performed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Keep paths portable in the recorded hash index.
portable={}
for name,value in hashes.items():
    path=Path(name)
    if path.is_relative_to(a.main_results.resolve()):relative='main_results/'+path.relative_to(a.main_results.resolve()).as_posix()
    else:relative='calibration_results/'+path.relative_to(a.calibration_results.resolve()).as_posix()
    portable[relative]=value
(a.output_root/'input_hashes.json').write_text(json.dumps(portable,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('SAVED_CALIBRATION_VERIFIED: 180 operating points and 30 selected-checkpoint losses; no training or model loading.')

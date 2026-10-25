"""Verify saved control predictions and checkpoints, and print a recording-ready summary.

This reads existing results only; it does not train or repeat the historical run.
"""
from pathlib import Path
import argparse,hashlib,json,csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path.cwd()/'fixed_target_results')
p.add_argument('--output',type=Path,default=Path.cwd()/'fixed_target_review')
a=p.parse_args();a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
methods=['fixed_target_r000','fixed_target_r010','fixed_target_r025','fixed_target_r050']
folders=sorted(a.root.glob('*_row_index_fixed'));assert len(folders)==6
rows=[];checks=0
for folder in folders:
    complete=json.loads((folder/'complete.json').read_text('utf-8'));assert complete['branches']==4
    mapping=json.loads((folder/'label_mapping.json').read_text('utf-8'))['global_labels']
    benign=next(int(k) for k,v in mapping.items() if v=='benign')
    direction,seed=folder.name.rsplit('_seed',1);seed=int(seed.split('_')[0])
    reference={}
    for method in methods:
        branch=folder/method;trained=json.loads((branch/'trained.json').read_text('utf-8'))
        assert hashlib.sha256((branch/'checkpoint.pt').read_bytes()).hexdigest()==trained['checkpoint_sha256']
        records=[json.loads(x) for x in (branch/'training.jsonl').read_text('utf-8').splitlines() if x.strip()]
        assert any(r.get('group')=='valid' and r.get('name')=='loss' for r in records)
        for domain in ['source','target','source_validation','target_validation']:
            with np.load(branch/f'{domain}_predictions.npz',allow_pickle=False) as z:
                y=z['y_true'];pred=z['y_pred'];logits=z['logits']
            assert np.isfinite(logits).all() and len(y)==len(pred)==len(logits)
            assert np.array_equal(pred,logits.argmax(1))
            if domain in reference:assert np.array_equal(y,reference[domain])
            else:reference[domain]=y.copy()
            _,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.unique(y),zero_division=0)
            normal=y==benign;attack=~normal
            values=dict(macro_f1_percent=100*float(f1.mean()),fpr_percent=100*float((pred[normal]!=benign).mean()),attack_recall_percent=100*float((pred[attack]!=benign).mean()))
            recorded=json.loads((branch/f'{domain}_metrics.json').read_text('utf-8'))
            assert int(recorded['n'])==len(y)
            for k,v in values.items():assert np.isclose(v,recorded[k],atol=1e-10,rtol=0),(folder.name,method,domain,k)
            checks+=1
            if domain in ['source','target']:rows.append(dict(direction=direction,seed=seed,method=method,domain=domain,n=len(y),**values))
    print('VERIFIED',folder.name,flush=True)
assert checks==96 and len(rows)==48
with (a.output/'recomputed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=defaultdict(list)
for r in rows:groups[r['direction'],r['method'],r['domain']].append(r)
means=[]
for (direction,method,domain),values in sorted(groups.items()):
    assert len(values)==3
    means.append(dict(direction=direction,method=method,domain=domain,**{k:float(np.mean([r[k] for r in values])) for k in ['macro_f1_percent','fpr_percent','attack_recall_percent']}))
with (a.output/'three_seed_means.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
for direction in sorted({r['direction'] for r in means}):
    print('\nDIRECTION',direction)
    print('METHOD                         SOURCE_F1 TARGET_F1 SOURCE_FPR TARGET_FPR')
    for method in methods:
        s=next(r for r in means if r['direction']==direction and r['method']==method and r['domain']=='source')
        t=next(r for r in means if r['direction']==direction and r['method']==method and r['domain']=='target')
        print(f"{method:30} {s['macro_f1_percent']:9.2f} {t['macro_f1_percent']:9.2f} {s['fpr_percent']:10.2f} {t['fpr_percent']:10.2f}")
report=dict(status='PASS',runs=6,branches=24,verified_predictions=checks,verified_test_rows=len(rows),checkpoints_hash_verified=24,training_performed=False,
            scope='Saved-result verification; not a recording or reconstruction of historical training.')
(a.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
print('\nRESULT_VERIFICATION_COMPLETE',json.dumps(report))

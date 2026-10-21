"""Attribute saved false positives and calibration changes to identical-input groups.

No training or threshold tuning: uses the archived 0.5% calibration rule.
"""
from pathlib import Path
from collections import defaultdict,Counter
import os,sys,json,csv,hashlib,argparse
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--main-results',type=Path,default=ROOT/'research_results')
parser.add_argument('--calibration-results',type=Path,default=ROOT/'research_results'/'fixed_calibration_20261019')
parser.add_argument('--source-root',type=Path,default=ROOT/'research_incremental_nids'/'src')
parser.add_argument('--output-root',type=Path,default=ROOT.parent/'数据与结果'/'测试组误报分解_20261021')
args=parser.parse_args()
OUT=args.output_root.resolve()
OUT.mkdir(parents=True,exist_ok=True)
MAIN=args.main_results.resolve();CAL=args.calibration_results.resolve()
SRC=args.source_root.resolve()
os.chdir(SRC);sys.path.insert(0,str(SRC))
from datasets.data_loader import get_loaders
from utils import seed_everything

def token(x):
    a=np.asarray(x,dtype='<f4').copy();a[a==0]=0
    assert np.isfinite(a).all()
    return hashlib.sha256(a.tobytes()).hexdigest()
def probabilities(z):
    z=np.asarray(z,dtype=np.float64);e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def macro(y,p):return 100*float(f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rows=[];group_records=[];class_rows=[]
for direction in ('iot_nidd_to_ton_iot','ton_iot_to_iot_nidd'):
    source,target=direction.split('_to_')
    for seed in (1,2,3):
        folder=MAIN/f'{direction}_seed{seed}_row_index_fixed'
        mapping=json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
        benign=next(int(k) for k,v in mapping.items() if v=='benign')
        seed_everything(seed)
        _,_,tests,_=get_loaders([source,target],1,0,0,64,0,
            validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
        for dataset,domain,loader in zip((source,target),('source','target'),tests):
            y=np.asarray(loader.dataset.labels)
            groups=defaultdict(list)
            for i,x in enumerate(loader.dataset.images):groups[token(x)].append(i)
            candidates=[(h,indices) for h,indices in groups.items() if benign in y[indices] and len(np.unique(y[indices]))>1]
            h,indices=max(candidates,key=lambda pair:sum(y[pair[1]]==benign))
            group=np.zeros(len(y),dtype=bool);group[indices]=True
            membership=OUT/'group_membership';membership.mkdir(exist_ok=True)
            np.savez_compressed(membership/f'{folder.name}_{domain}.npz',indices=np.asarray(indices,dtype=np.int64),y_true=y[indices],input_sha256=np.asarray(h),test_n=np.asarray(len(y)))
            normal=y==benign;gn=group&normal
            group_records.append(dict(run=folder.name,dataset=dataset,domain=domain,
                input_sha256=h,n=len(indices),labels={mapping[str(int(k))]:int(n) for k,n in Counter(y[indices]).items()},
                benign_test_total=int(normal.sum()),group_benign_total=int(gn.sum()),
                max_fpr_contribution_points=100*float(gn.sum()/normal.sum())))
            for method in ('source','ft','ft_mem','replay','weighted','combined'):
                with np.load(folder/method/f'{domain}_predictions.npz',allow_pickle=False) as z:
                    assert np.array_equal(y,z['y_true'])
                    raw=z['y_pred'];ztest=z['logits']
                assert np.array_equal(raw,ztest.argmax(1))
                fp=(raw!=benign)&normal;gfp=fp&group
                row=dict(run=folder.name,direction=direction,seed=seed,dataset=dataset,domain=domain,method=method,
                    group_sha256=h,group_n=len(indices),group_benign_n=int(gn.sum()),benign_test_n=int(normal.sum()),
                    raw_fp=int(fp.sum()),raw_group_fp=int(gfp.sum()),raw_fpr_percent=100*float(fp.sum()/normal.sum()),
                    raw_group_fpr_points=100*float(gfp.sum()/normal.sum()),
                    raw_group_predictions=';'.join(f'{mapping[str(int(k))]}:{n}' for k,n in sorted(Counter(raw[group]).items())),
                    raw_macro_f1=macro(y,raw),threshold=None,cal_fp=None,cal_group_fp=None,
                    cal_fpr_percent=None,cal_group_fpr_points=None,cal_macro_f1=None,
                    changed_predictions=None,changed_in_group=None,group_shapley_macro_delta=None,other_shapley_macro_delta=None,
                    cal_group_predictions=None)
                pred=None
                if method!='source':
                    vp=CAL/f'{direction}_seed{seed}'/method/'validation_predictions.npz'
                    with np.load(vp,allow_pickle=False) as v:
                        pb=probabilities(v['logits'])[v['y_true']==benign,benign]
                    values,counts=np.unique(pb,return_counts=True)
                    below=np.r_[0,np.cumsum(counts)[:-1]]
                    threshold=float(values[below<=int(np.floor(.005*len(pb)))][-1])
                    pt=probabilities(ztest);pa=pt.copy();pa[:,benign]=-1
                    pred=pa.argmax(1);pred[pt[:,benign]>=threshold]=benign
                    cfp=(pred!=benign)&normal;cgfp=cfp&group
                    only_group=raw.copy();only_group[group]=pred[group]
                    only_other=pred.copy();only_other[group]=raw[group]
                    group_delta=.5*((macro(y,only_group)-macro(y,raw))+(macro(y,pred)-macro(y,only_other)))
                    full_delta=macro(y,pred)-macro(y,raw)
                    row.update(threshold=threshold,cal_fp=int(cfp.sum()),cal_group_fp=int(cgfp.sum()),
                        cal_fpr_percent=100*float(cfp.sum()/normal.sum()),cal_group_fpr_points=100*float(cgfp.sum()/normal.sum()),
                        cal_macro_f1=macro(y,pred),changed_predictions=int((pred!=raw).sum()),
                        changed_in_group=int(((pred!=raw)&group).sum()),group_shapley_macro_delta=group_delta,
                        other_shapley_macro_delta=full_delta-group_delta,
                        cal_group_predictions=';'.join(f'{mapping[str(int(k))]}:{n}' for k,n in sorted(Counter(pred[group]).items())))
                rows.append(row)
                for k in np.unique(y[group]):
                    true=y==k;ingroup=true&group
                    class_rows.append(dict(run=folder.name,dataset=dataset,domain=domain,method=method,label=mapping[str(int(k))],
                        class_n=int(true.sum()),group_class_n=int(ingroup.sum()),raw_tp=int(((raw==k)&true).sum()),
                        raw_group_tp=int(((raw==k)&ingroup).sum()),
                        cal_tp=None if pred is None else int(((pred==k)&true).sum()),
                        cal_group_tp=None if pred is None else int(((pred==k)&ingroup).sum())))
        print('GROUP_AUDIT',folder.name,flush=True)
write_csv('group_prediction_attribution.csv',rows)
write_csv('group_class_recall.csv',class_rows)
(OUT/'groups.json').write_text(json.dumps(group_records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(dict(match='Exact float32 four-field ten-packet input, treating signed zeros as equal; labels excluded from match.',
    group_selection='Largest benign-containing multi-label identical-input test group, selected independently of predictions.',
    calibration='Archived 0.5% target-validation budget; same float64 softmax and tied-probability boundary.',
    attribution='FPR contributions use the full-domain benign denominator. Two-block Shapley decomposition assigns the exact Macro-F1 change between group-only and other-only prediction replacements; this is arithmetic attribution, not a causal training intervention.',
    training_performed=False),indent=2)+'\n',encoding='utf-8')
print('COMPLETE',len(rows),'method-domain rows',len(class_rows),'class rows',flush=True)

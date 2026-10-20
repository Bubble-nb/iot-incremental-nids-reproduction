"""固定验证集误报预算，复算现有模型的分类/告警运行点；不重新训练。"""
from pathlib import Path
from datetime import datetime
import os, sys, json, hashlib, csv
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'research_results' / 'fixed_calibration_20261019'
OUT.mkdir(exist_ok=True)
RULE = dict(version=1, primary_fpr_budget=0.005, sensitivity_budgets=[0.01, 0.05],
    threshold='largest observed benign validation probability with empirical count(p<threshold)/n <= budget',
    prediction='p(benign)>=threshold => benign; otherwise highest-probability non-benign label',
    arithmetic='NumPy float64 softmax, same boundary in validation and test',
    methods=['ft','ft_mem','replay','weighted','combined'], seeds=[1,2,3],
    directions=[['iot_nidd','ton_iot'],['ton_iot','iot_nidd']],
    evidence_scope='Post-hoc fixed-rule cross-run assessment. Existing test results were previously inspected; not independent confirmatory evidence.',
    training='No retraining. Reload saved checkpoints only to recover validation logits; saved test logits unchanged.')
freeze = OUT / 'frozen_protocol.json'
if freeze.exists():
    assert json.loads(freeze.read_text('utf-8'))['rule'] == RULE
else:
    freeze.write_text(json.dumps(dict(frozen_at=datetime.now().isoformat(),rule=RULE),ensure_ascii=False,indent=2),encoding='utf-8')
src = ROOT / 'research_incremental_nids' / 'src'
os.chdir(src); sys.path.insert(0,str(src))
from datasets.data_loader import get_loaders
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'

def probs(z):
    z=np.asarray(z,dtype=np.float64); e=np.exp(z-z.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)

def metrics(y,p,b):
    return dict(macro_f1=float(f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0)),
        fpr=float(np.mean(p[y==b]!=b)),attack_recall=float(np.mean(p[y!=b]!=b)),n=len(y),n_benign=int(sum(y==b)))

rows=[]; hashes={}
for source,target in RULE['directions']:
  for seed in RULE['seeds']:
    folder=ROOT/'research_results'/f'{source}_to_{target}_seed{seed}_row_index_fixed'
    protocol=json.loads((folder/'protocol.json').read_text('utf-8'))
    mapping=json.loads((folder/'label_mapping.json').read_text('utf-8'))
    b=next(int(k) for k,v in mapping['global_labels'].items() if v=='benign')
    seed_everything(seed)
    train,valid,test,taskcla=get_loaders([source,target],1,0,0,64,0,validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
    assert {str(k):v for k,v in Counter(valid[1].dataset.labels).items()} == protocol['validation_counts'][1]
    for method in RULE['methods']:
      checkpoint=folder/method/'checkpoint.pt'
      model=LLL_Net(Lopez17CNN(num_pkts=10,num_fields=4))
      for _,n in taskcla: model.add_head(n)
      model.to(device); model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)); model.eval()
      zs=[]; ys=[]
      with torch.no_grad():
        for x,y in valid[1]:
          zs.append(torch.cat(model(x.to(device)),dim=1).cpu().numpy()); ys.extend(y.tolist())
      z=np.concatenate(zs); y=np.asarray(ys); pv=probs(z); benign=pv[y==b,b]
      runout=OUT/f'{source}_to_{target}_seed{seed}'/method; runout.mkdir(parents=True,exist_ok=True)
      np.savez_compressed(runout/'validation_predictions.npz',logits=z,y_true=y)
      hashes[str(checkpoint.relative_to(ROOT))]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
      val_losses=[r for r in map(json.loads,(folder/method/'training.jsonl').read_text('utf-8').splitlines()) if r.get('group')=='valid' and r.get('name')=='loss']
      val_ce=float((-np.log(pv[np.arange(len(y)),y])).mean())
      assert abs(val_ce-min(r['value'] for r in val_losses))<1e-5, (method,seed,val_ce)
      thresholds={}
      for alpha in [RULE['primary_fpr_budget']]+RULE['sensitivity_budgets']:
        values,counts=np.unique(benign,return_counts=True)
        below=np.r_[0,np.cumsum(counts)[:-1]]
        threshold=float(values[below<=int(np.floor(alpha*len(benign)))][-1])
        thresholds[alpha]=threshold
      for domain in ['source','target']:
        predfile=folder/method/f'{domain}_predictions.npz'
        d=np.load(predfile); pt=probs(d['logits']); yt=d['y_true']
        hashes[str(predfile.relative_to(ROOT))]=hashlib.sha256(predfile.read_bytes()).hexdigest()
        base=metrics(yt,d['y_pred'],b)
        for alpha,threshold in thresholds.items():
          pa=pt.copy(); pa[:,b]=-1; prediction=pa.argmax(1); prediction[pt[:,b]>=threshold]=b
          m=metrics(yt,prediction,b)
          rows.append(dict(source=source,target=target,seed=seed,method=method,domain=domain,budget=alpha,
             benign_label=b,threshold=threshold,val_benign_n=len(benign),val_fpr=float(np.mean(benign<threshold)),
             **m,**{'baseline_'+k:v for k,v in base.items() if k in ['macro_f1','fpr','attack_recall']}))
      print('CALIBRATED',source,target,seed,method,thresholds[0.005],flush=True)
with (OUT/'operating_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
for source,target in RULE['directions']:
    for method in ['ft_mem','combined']:
        rr=[r for r in rows if r['source']==source and r['method']==method and r['domain']=='target' and r['budget']==.005]
        print('SUMMARY',source,method,json.dumps({k:[round(100*r[k],3) for r in rr] for k in ['macro_f1','fpr','attack_recall']},ensure_ascii=False),flush=True)
print('COMPLETE',OUT,flush=True)

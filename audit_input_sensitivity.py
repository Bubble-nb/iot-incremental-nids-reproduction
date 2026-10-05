"""Recompute fixed-checkpoint metrics after excluding seen model inputs.

No fitting, threshold selection, or test-guided model selection is performed.
Runs from this directory after the author research copy and results are present.
"""
from pathlib import Path
import os, sys, json, hashlib, csv
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / '数据与结果' / '重复输入敏感性'
OUT.mkdir(parents=True, exist_ok=True)
src = ROOT / 'research_incremental_nids' / 'src'
os.chdir(src)
sys.path.insert(0, str(src))
from datasets.data_loader import get_loaders
from networks.lopez17cnn import Lopez17CNN
from networks.network import LLL_Net
from utils import seed_everything
torch.set_num_threads(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def token(x):
    a = np.asarray(x, dtype='<f4').copy()
    a[a == 0] = 0  # Treat +0 and -0 as equal numeric inputs.
    assert np.isfinite(a).all()
    return hashlib.sha256(a.tobytes()).hexdigest()

def dump_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

metrics, classes, checks, resources = [], [], [], []
test_identities = {}
for folder in sorted((ROOT / 'research_results').glob('*seed[123]_row_index_fixed')):
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    seed = protocol['training_seed']
    names = folder.name.split('_to_')
    source = names[0]; target = names[1].split('_seed')[0]
    seed_everything(seed)
    train, valid, test, taskcla = get_loaders([source, target], 1, 0, 0, 64, 0,
        validation=.1, num_pkts=10, fields=['PL','IAT','DIR','WIN'], seed=1)
    mapping = json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    benign = next(int(k) for k,v in mapping.items() if v == 'benign')
    pools = [set(token(x) for x in list(t.dataset.images) + list(v.dataset.images))
             for t,v in zip(train, valid)]
    union = pools[0] | pools[1]
    for method in ['ft_mem', 'replay', 'weighted', 'combined']:
        model = LLL_Net(Lopez17CNN(num_pkts=10, num_fields=4))
        model.add_head(taskcla[0][1]); model.add_head(taskcla[1][1])
        model.load_state_dict(torch.load(folder/method/'checkpoint.pt', map_location='cpu', weights_only=True))
        model.to(device).eval()
        complete = json.loads((folder/method/'complete.json').read_text(encoding='utf-8'))
        log = [json.loads(line) for line in (folder/method/'training.jsonl').read_text().splitlines()]
        val = [r for r in log if r.get('group') == 'valid' and r.get('name') == 'loss']
        best = min(val, key=lambda r:r['value'])
        resources.append(dict(run=folder.name, method=method,
            elapsed_seconds=complete['elapsed_seconds'], best_epoch=best['iter'],
            last_epoch=max(r['iter'] for r in val)))
        for d, (domain,loader) in enumerate(zip(['source','target'],test)):
            saved = np.load(folder/method/f'{domain}_predictions.npz')
            y = np.asarray(loader.dataset.labels)
            assert np.array_equal(y,saved['y_true']), (folder.name, method, domain, 'label order')
            logits = []
            with torch.no_grad():
                for x,_ in loader:
                    logits.append(torch.cat(model(x.to(device)),dim=1).cpu().numpy())
            recomputed = np.concatenate(logits)
            error = float(np.max(np.abs(recomputed-saved['logits'])))
            assert np.allclose(recomputed,saved['logits'],rtol=1e-5,atol=1e-5), (folder.name,method,domain,error)
            pred = saved['y_pred']
            assert np.array_equal(pred,recomputed.argmax(1))
            checks.append(dict(run=folder.name,method=method,domain=domain,n=len(y),
                label_order_matches=True,checkpoint_predictions_match=True,max_logit_difference=error))
            fingerprints=[token(x) for x in loader.dataset.images]
            semantic_items=sorted(h+'|'+mapping[str(int(k))] for h,k in zip(fingerprints,y))
            identity=hashlib.sha256('\n'.join(semantic_items).encode()).hexdigest()
            test_identities.setdefault(f'{source}_to_{target}/{domain}',set()).add(identity)
            checks[-1]['test_input_semantic_label_multiset_sha256']=identity
            masks = {'all':np.ones(len(y),dtype=bool),
                     'unseen_within_domain':np.array([h not in pools[d] for h in fingerprints]),
                     'unseen_in_either_domain':np.array([h not in union for h in fingerprints])}
            original_classes=np.unique(y)
            for subset,mask in masks.items():
                yt,yp=y[mask],pred[mask]; labels=np.unique(yt)
                assert len(yt)>0
                _,rec,f1,sup=precision_recall_fscore_support(yt,yp,labels=labels,zero_division=0)
                normal=yt==benign; attack=~normal
                fpr=float((yp[normal]!=benign).mean()) if normal.any() else None
                recall=float((yp[attack]!=benign).mean()) if attack.any() else None
                metrics.append(dict(run=folder.name,seed=seed,direction=f'{source}_to_{target}',method=method,
                    domain=domain,subset=subset,n=len(yt),retained_percent=100*len(yt)/len(y),
                    surviving_classes=len(labels),original_classes=len(original_classes),
                    absent_classes=';'.join(mapping[str(int(k))] for k in original_classes if k not in labels),
                    macro_f1_percent=100*float(f1.mean()),fpr_percent=None if fpr is None else 100*fpr,
                    attack_recall_percent=None if recall is None else 100*recall,
                    benign_n=int(normal.sum()),attack_n=int(attack.sum())))
                byclass={int(k):(float(f),float(r),int(n)) for k,f,r,n in zip(labels,f1,rec,sup)}
                for k in original_classes:
                    f,r,n=byclass.get(int(k),(None,None,0))
                    classes.append(dict(run=folder.name,method=method,domain=domain,subset=subset,
                        label=mapping[str(int(k))],support=n,f1_percent=None if f is None else 100*f,
                        recall_percent=None if r is None else 100*r))
        print(folder.name,method,'PASS',flush=True)
dump_csv('subset_metrics.csv',metrics)
dump_csv('per_class_metrics.csv',classes)
dump_csv('training_resources.csv',resources)
assert all(len(v)==1 for v in test_identities.values()), 'Test membership changed between seeds.'
(OUT/'alignment_checks.json').write_text(json.dumps(dict(device=device,checks=checks,
    test_membership_fixed_across_seeds=True,
    exclusion='Numeric float32 model input equality; label not part of fingerprint; train plus validation pools.',
    limitation='Post hoc fixed-checkpoint sensitivity, not retraining or independent-network validation.'),indent=2),encoding='utf-8')
print('COMPLETE',len(checks),'checkpoint/domain alignments;',len(metrics),'metric rows',flush=True)

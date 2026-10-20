"""Count exact feature-vector collisions with conflicting labels in fixed samples."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib, json, os, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'数据与结果'/'实验设计复核'
src=ROOT/'research_incremental_nids'/'src'
os.chdir(src);sys.path.insert(0,str(src))
from datasets.data_loader import get_loaders
from utils import seed_everything

def token(x):
    data=np.asarray(x,dtype='<f4').copy()
    data[data==0]=0
    assert np.isfinite(data).all()
    return hashlib.sha256(data.tobytes()).hexdigest()

output=[]
for direction in ('iot_nidd_to_ton_iot','ton_iot_to_iot_nidd'):
    folder=ROOT/'research_results'/f'{direction}_seed1_row_index_fixed'
    source,target=direction.split('_to_')
    seed_everything(1)
    train,valid,test,_=get_loaders([source,target],1,0,0,64,0,
        validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
    names=json.loads((folder/'label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    for domain,d in (('source',0),('target',1)):
        saved=np.load(folder/'ft_mem'/f'{domain}_predictions.npz')
        assert np.array_equal(np.asarray(test[d].dataset.labels),saved['y_true']), (direction,domain,'label alignment')
        seen=defaultdict(set)
        train_rows=0
        for loader in (train[d],valid[d]):
            for x,y in zip(loader.dataset.images,loader.dataset.labels):
                seen[token(x)].add(int(y));train_rows+=1
        colliding={key:values for key,values in seen.items() if len(values)>1}
        test_rows=0; matched=0; conflicting=0; same_label=0
        pairs=Counter();mitm_conflicts=0
        for x,y in zip(test[d].dataset.images,test[d].dataset.labels):
            test_rows+=1
            labels=seen.get(token(x))
            if labels is None:continue
            matched+=1
            if int(y) in labels:same_label+=1
            others=labels-{int(y)}
            if others:
                conflicting+=1
                for other in others:
                    pairs[(names[str(int(y))],names[str(other)])]+=1
                    if 'mitm' in names[str(int(y))] or 'mitm' in names[str(other)]:mitm_conflicts+=1
        output.append(dict(direction=direction,domain=domain,
            training_validation_rows=train_rows,distinct_training_validation_inputs=len(seen),
            training_validation_inputs_with_multiple_labels=len(colliding),
            test_rows=test_rows,test_exact_input_matches=matched,
            matched_test_rows_with_same_label=same_label,
            matched_test_rows_with_conflicting_label=conflicting,
            mitm_conflict_pairs=mitm_conflicts,
            top_conflict_pairs=[dict(test_label=a,training_label=b,rows=n) for (a,b),n in pairs.most_common(10)]))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'重复输入标签冲突.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False,indent=2))

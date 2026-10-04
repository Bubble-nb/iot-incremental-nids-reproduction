from pathlib import Path
import os, sys, json, hashlib
import numpy as np
root=Path(__file__).resolve().parent
src=root/'research_incremental_nids/src'
os.chdir(src)
sys.path.insert(0,str(src))
from utils import seed_everything
from datasets.data_loader import get_loaders
records=[]
for seed in [1,2,3]:
    seed_everything(seed)
    train,valid,test,taskcla=get_loaders(['iot_nidd','ton_iot'],1,0,0,64,0,
        validation=.1,num_pkts=10,fields=['PL','IAT','DIR','WIN'],seed=1)
    mapping=json.loads((root/f'research_results/iot_nidd_to_ton_iot_seed{seed}_row_index_fixed/label_mapping.json').read_text(encoding='utf-8'))['global_labels']
    row=dict(seed=seed,domains=[])
    for domain,validation_loader,test_loader in zip(['source','target'],valid,test):
        digests={}
        for split,loader in [('validation',validation_loader),('test',test_loader)]:
            items=[]
            for image,label in zip(loader.dataset.images,loader.dataset.labels):
                token=mapping[str(int(label))].encode()+b'\0'+np.asarray(image,dtype='<f4').tobytes()
                items.append(hashlib.sha256(token).hexdigest())
            digests[split]=dict(n=len(items),feature_label_multiset_sha256=hashlib.sha256('\n'.join(sorted(items)).encode()).hexdigest())
        row['domains'].append(dict(domain=domain,**digests))
    records.append(row)
result=dict(records=records,note='Order-independent multiset of float32 model inputs and semantic class names; does not establish raw flow identity.')
for idx,domain in enumerate(['source','target']):
    assert len({r['domains'][idx]['test']['feature_label_multiset_sha256'] for r in records})==1
    result[domain+'_validation_varies']=len({r['domains'][idx]['validation']['feature_label_multiset_sha256'] for r in records})>1
(root/'research_results/repeat_validation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print({key:value for key,value in result.items() if key!='records'})

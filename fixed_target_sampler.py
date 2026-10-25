"""Replacement sampling with an identical pre-batching target stream per epoch."""
import math,hashlib
import numpy as np
from torch.utils.data import Sampler

def batch_counts(n,ratio):
    assert n>0 and 0<=ratio<=.5
    counts=[];left=n;i=0
    while left:
        old=math.floor((i+1)*64*ratio+1e-9)-math.floor(i*64*ratio+1e-9)
        new=min(64-old,left);counts.append((new,old));left-=new;i+=1
    return counts

class FixedTargetBatches(Sampler):
    def __init__(self,n,m,ratio,seed,folder=None):
        self.n=n;self.m=m;self.seed=seed;self.counts=batch_counts(n,ratio);self.epoch=0;self.folder=folder
    def __len__(self):return len(self.counts)
    def streams(self,epoch):
        new=np.random.default_rng(self.seed+100000+epoch).integers(self.n,size=self.n,dtype=np.int64)
        old=np.random.default_rng(self.seed+200000+epoch).integers(self.m,size=sum(o for _,o in self.counts),dtype=np.int64)+self.n
        return new,old
    def __iter__(self):
        import json
        self.epoch+=1;new,old=self.streams(self.epoch)
        if self.folder is not None:
            record=dict(epoch=self.epoch,target_draws=len(new),memory_draws=len(old),updates=len(self),
                unique_target_samples=len(np.unique(new)),target_stream_sha256=hashlib.sha256(new.astype('<i8').tobytes()).hexdigest())
            with (self.folder/'exposure.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        rng=np.random.default_rng(self.seed+300000+self.epoch);i=j=0
        for nn,no in self.counts:
            batch=np.r_[new[i:i+nn],old[j:j+no]];i+=nn;j+=no
            yield batch[rng.permutation(len(batch))].tolist()
        assert i==self.n and j==len(old)

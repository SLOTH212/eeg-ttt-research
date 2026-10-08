from pathlib import Path
import json, numpy as np, pyedflib, time
from scipy.signal import welch
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/pilot.json').read_text())
assert (ROOT/'phase2/evaluation_lock.json').exists()
manifest=json.loads((ROOT/'phase2/cohort_manifest.json').read_text())['files']
cfg['subject_ids']=list(range(20,39))
(ROOT/'phase2/processed').mkdir(exist_ok=True)
audit=[]
for sid in cfg['subject_ids']:
    recs=[r for r in manifest if r['subject']==sid]
    p=next(ROOT/'phase2/raw'/Path(r['path']).name for r in recs if r['path'].endswith('-PSG.edf'))
    h=next(ROOT/'phase2/raw'/Path(r['path']).name for r in recs if r['path'].endswith('-Hypnogram.edf'))
    while not p.exists() or not h.exists():time.sleep(2)
    with pyedflib.EdfReader(str(p)) as f:
        idx=[f.getSignalLabels().index(c) for c in cfg['channels']]
        assert all(f.getSampleFrequency(i)==100 for i in idx)
        units=[f.getPhysicalDimension(i) for i in idx]
        assert all(u.strip() in ('uV','µV') for u in units),units
        start=f.getStartdatetime(); raw=np.stack([f.readSignal(i) for i in idx]).astype('float32')
    n=raw.shape[1]//3000
    x=raw[:,:n*3000].reshape(2,n,3000).transpose(1,0,2).copy()
    # No future windows used: features and centering confined to current 30-second window.
    x-=x.mean(axis=-1,keepdims=True);x=np.clip(x/100.,-20,20)
    with pyedflib.EdfReader(str(h)) as f:
        delta=(f.getStartdatetime()-start).total_seconds();on,dur,desc=f.readAnnotations()
    y=np.full(n,-1,dtype='int64'); assigned=np.zeros(n,dtype=bool)
    for o,d,s in zip(on,dur,desc):
        o+=delta
        a=max(0,int(np.ceil(o/30)));b=min(n,int(np.floor((o+d)/30)))
        if b>a:
            assert not assigned[a:b].any(),(sid,o,d)
            y[a:b]=cfg['labels'].get(s,-1);assigned[a:b]=True
    f,power=welch(x,fs=100,nperseg=400,noverlap=200,axis=-1)
    edges=[.5,2,4,6,8,10,12,14,16,20,25,30,40]
    bands=np.stack([power[:,:,(f>=a)&(f<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],axis=-1)
    logpower=np.log(bands+1e-10)
    rel=np.log((bands+1e-10)/(bands.sum(-1,keepdims=True)+1e-10))
    sd=x.std(-1); diff=np.diff(x,axis=-1).std(-1)/(sd+1e-6)
    feats=np.concatenate([logpower.reshape(n,-1),rel.reshape(n,-1),np.log(sd+1e-6),diff],axis=1).astype('float32')
    np.save(ROOT/f'phase2/processed/s{sid:02d}_x.npy',x)
    np.savez(ROOT/f'phase2/processed/s{sid:02d}_meta.npz',y=y,features=feats)
    rec={'subject':sid,'record':p.name,'windows':n,'counts':np.bincount(y[y>=0],minlength=5).tolist(),'unscored':int((y<0).sum()),'annotation_offset_seconds':delta,'units':units,'finite':bool(np.isfinite(x).all() and np.isfinite(feats).all())}
    assert rec['finite'];audit.append(rec);print(json.dumps(rec),flush=True)
(ROOT/'phase2/data_audit.json').write_text(json.dumps(audit,indent=2))
print('PREPARE COMPLETE',flush=True)

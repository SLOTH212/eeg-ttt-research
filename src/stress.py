from adaptation import *
from controllers import fit_gate
from scipy.signal import welch
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent
m,ck=load_model(path);kind=ck['kind'];split=CFG['folds'][ck['fold']];lr=json.loads((out/'selection.json').read_text())['lr'];sup=json.loads((out/'supplement.json').read_text());alpha=sup['smooth_alpha']
gate=fit_gate(json.loads((out/'controller_diagnostics.json').read_text())+json.loads((out/'source_replay_diagnostics.json').read_text()))
def features(x):
 f,p=welch(x,fs=100,nperseg=400,noverlap=200,axis=-1);edges=[.5,2,4,6,8,10,12,14,16,20,25,30,40]
 bands=np.stack([p[:,:,(f>=a)&(f<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],axis=-1)
 logpower=np.log(bands+1e-10);rel=np.log((bands+1e-10)/(bands.sum(-1,keepdims=True)+1e-10));sd=x.std(-1);diff=np.diff(x,axis=-1).std(-1)/(sd+1e-6)
 return np.concatenate([logpower.reshape(len(x),-1),rel.reshape(len(x),-1),np.log(sd+1e-6),diff],axis=1).astype('float32')
rows=[]
for sid in split['evaluation'][:2]:
 raw,y=load_subject(sid,'eegnet');raw=np.array(raw);original=features(raw[:20]);cached=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'][:20];assert np.allclose(original,cached,atol=1e-6)
 rng=np.random.default_rng(20261002+sid);n=len(raw);stratum=np.full(n,'clean',dtype='<U20');segments=[]
 for i,typ in enumerate(['amplitude_x2','noise_50uv','channel_dropout']):
  start=(i+1)*n//4;end=start+80
  if typ=='amplitude_x2':raw[start:end]*=2
  elif typ=='noise_50uv':raw[start:end]+=rng.normal(0,.5,size=raw[start:end].shape).astype('float32')
  else:raw[start:end,0,:]=0
  stratum[start:end]=typ;stratum[end:end+80]='recovery';segments.append({'type':typ,'start':start,'end':end,'recovery_end':end+80})
 raw=np.clip(raw,-20,20);hf=features(raw);x=hf if kind=='spectral' else raw
 preds={'frozen':predict(m,x)};preds['smooth_tuned']=smooth(preds['frozen'],alpha)
 for name,mode,g in [('continuous','continuous',None),('reset_periodic','reset_periodic',None),('refined_controller','controller',gate)]:
  preds[name],_,_=stream(m,x,mode,lr,g,hf)
 for name,pr in preds.items():
  rows.append({'subject':sid,'method':name,'stratum':'all',**metrics(y,pr)})
  for s in sorted(set(stratum)):
   mask=stratum==s;rows.append({'subject':sid,'method':name,'stratum':str(s),**metrics(y[mask],pr[mask])})
 np.savez_compressed(out/f'stress_predictions_s{sid:02d}.npz',y=y,stratum=stratum,**preds)
 print('STRESS',sid,segments,{k:metrics(y,v)['macro_f1'] for k,v in preds.items()},flush=True)
(out/'stress.json').write_text(json.dumps({'results':rows,'protocol':'Synthetic perturbations at fixed timeline fractions, no labels used for locations. 80-window blocks and 80-window recovery. Only seed17, first two evaluation subjects per fold. Not clinical shifts.'},indent=2));print('STRESS COMPLETE',flush=True)

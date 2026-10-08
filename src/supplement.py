from adaptation import *
from controllers import fit_gate
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent
m,ck=load_model(path);split=CFG['folds'][ck['fold']];kind=ck['kind'];lr=json.loads((out/'selection.json').read_text())['lr']
train=json.loads((out/'controller_diagnostics.json').read_text());gate=fit_gate(train)
def subj(sid):
 x,y=load_subject(sid,kind);hf=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'];return x,y,hf
# One on-policy source replay, then refit; all extra labels stay in the controller split.
replay=[]
for sid in split['controller_train']:
 x,y,hf=subj(sid);_,ds,tm=stream(m,x,'controller',lr,gate,hf,True,sid);replay.extend(score_diagnostics(ds,y,sid,'controller_replay'))
 print('REPLAY',sid,tm['seconds'],flush=True)
refined=fit_gate(train+replay)
# Source-only tuning of causal smoothing; no target-dependent thresholds.
val={s:subj(s)[:2] for s in split['base_val']};ps={s:predict(m,x) for s,(x,y) in val.items()}
scores=[{'alpha':a,'f1':float(np.mean([metrics(val[s][1],smooth(ps[s],a))['macro_f1'] for s in val]))} for a in [.5,.8,.95]]
alpha=max(scores,key=lambda v:v['f1'])['alpha']
# Determine matched random commit rate from source replay, not target actions.
source_features=[r['features'] for r in replay if r['h']==4];commit_rate=float(np.mean([gate(f)==1 for f in source_features]));reset_rate=float(np.mean([gate(f)==2 for f in source_features]))
thresholds=np.quantile([r['features'][1] for r in train if r['h']==4],[.25,.5,.75])
threshold_scores=[]
for threshold in thresholds:
 ss=[]
 for sid in split['base_val']:
  x,y,hf=subj(sid);pr,_,_=stream(m,x,'controller',lr,lambda f:int(f[1]<threshold),hf);ss.append(metrics(y,pr)['macro_f1'])
 threshold_scores.append({'threshold':float(threshold),'f1':float(np.mean(ss))})
grad_threshold=max(threshold_scores,key=lambda v:v['f1'])['threshold']
results=[]
for sid in split['evaluation']:
 x,y,hf=subj(sid);rng=np.random.default_rng(ck['seed']*100+sid)
 def random_gate(f):
  u=rng.random();return 1 if u<commit_rate else (2 if u<commit_rate+reset_rate else 0)
 pr,_,tm=stream(m,x,'controller',lr,refined,hf);preds={'refined_controller':pr};timings={'refined_controller':tm}
 for name,g in [('matched_random',random_gate),('low_gradient',lambda f:int(f[1]<grad_threshold)),('no_reset_controller',lambda f:int(gate.values(f)[0,0]>0))]:
  pp,_,tt=stream(m,x,'controller',lr,g,hf);preds[name]=pp;timings[name]=tt
 t=time.perf_counter();preds['causal_bn_corrected']=causal_bn(m,x);timings['causal_bn_corrected']={'seconds':time.perf_counter()-t}
 frozen=np.load(out/f'predictions_s{sid:02d}.npz')['frozen'];preds['smooth_tuned']=smooth(frozen,alpha)
 np.savez_compressed(out/f'supplement_predictions_s{sid:02d}.npz',y=y,**preds)
 for name,pp in preds.items():results.append({'subject':sid,'method':name,**metrics(y,pp),**timings.get(name,{})})
 print('SUPPLEMENT',sid,{name:metrics(y,pp)['macro_f1'] for name,pp in preds.items()},flush=True)
(out/'supplement.json').write_text(json.dumps({'results':results,'smooth_selection':scores,'smooth_alpha':alpha,'source_commit_rate':commit_rate,'source_reset_rate':reset_rate,'gradient_selection':threshold_scores,'gradient_threshold':grad_threshold,'replay_rows':len(replay)},indent=2))
(out/'source_replay_diagnostics.json').write_text(json.dumps(replay));print('SUPPLEMENT COMPLETE',flush=True)

from adaptation import *
from controllers import fit_gate
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent;m,ck=load_model(path);split=CFG['folds'][ck['fold']];kind=ck['kind'];old=json.loads((out/'selection.json').read_text());scores=list(old['source_validation'])
def subj(sid):
 x,y=load_subject(sid,kind);hf=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'];return x,y,hf
frozen_score=float(np.mean([metrics(y,predict(m,x))['macro_f1'] for x,y,hf in [subj(s) for s in split['base_val']]]))
for lr in [1e-6,1e-5]:
 per=[]
 for sid in split['base_val']:
  x,y,hf=subj(sid);pr,_,_=stream(m,x,'continuous',lr,history_features=hf);per.append(metrics(y,pr)['macro_f1'])
 scores.append({'lr':lr,'macro_f1':float(np.mean(per))});print('EXTENDED_LR',lr,per,flush=True)
best=max(scores,key=lambda r:r['macro_f1']);lr=best['lr'];retuned=lr!=old['lr'];print('EXTENDED_SELECTION',best,'frozen',frozen_score,flush=True)
if retuned:
 rows=[]
 for sid in split['controller_train']:
  x,y,hf=subj(sid)
  for mode in ['continuous','periodic']:
   _,ds,_=stream(m,x,mode,lr,history_features=hf,collect=True,subject=sid);rows.extend(score_diagnostics(ds,y,sid,mode))
 gate=fit_gate(rows)
else:gate=fit_gate(json.loads((out/'controller_diagnostics.json').read_text()))
results=[]
for sid in split['evaluation']:
 x,y,hf=subj(sid);preds={}
 for rate,name in [(1e-6,'lr_1e6'),(1e-5,'lr_1e5'),(lr,'extended_lr_continuous')]:
  # Reuse identical rate trajectories where possible.
  cached=next((preds[n] for r,n in [(1e-6,'lr_1e6'),(1e-5,'lr_1e5')] if r==rate and n in preds),None)
  if cached is None:pr,_,_=stream(m,x,'continuous',rate,history_features=hf)
  else:pr=cached
  preds[name]=pr
 if retuned:preds['retuned_future_controller'],_,_=stream(m,x,'controller',lr,gate,hf)
 else:preds['retuned_future_controller']=np.load(out/f'predictions_s{sid:02d}.npz')['future_controller']
 for name,pr in preds.items():results.append({'subject':sid,'method':name,**metrics(y,pr)})
 np.savez_compressed(out/f'lr_predictions_s{sid:02d}.npz',y=y,**preds);print('LR_EVAL',sid,{k:metrics(y,v)['macro_f1'] for k,v in preds.items()},flush=True)
(out/'lr_sensitivity.json').write_text(json.dumps({'results':results,'source_scores':scores,'source_frozen_f1':frozen_score,'selected_lr':lr,'old_lr':old['lr'],'controller_refitted':retuned,'source_prefers_frozen':frozen_score>best['macro_f1'],'notes':'Exploratory expansion to 1e-6 and 1e-5 after source-domain grid-edge failure, source labels only; retuned controller uses same source mixed trajectories and ridge settings without extra replay.'},indent=2));print('LR SENSITIVITY COMPLETE',flush=True)

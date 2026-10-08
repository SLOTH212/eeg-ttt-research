from adaptation import *
from controllers import fit_gate
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent;m,ck=load_model(path);kind=ck['kind'];split=CFG['folds'][ck['fold']];lr=json.loads((out/'selection.json').read_text())['lr']
rows=json.loads((out/'controller_diagnostics.json').read_text())+json.loads((out/'source_replay_diagnostics.json').read_text())
gates={typ:fit_gate(rows,typ) for typ in ['future','confidence','current']}
def identity_gate(g):
 def choose(f):
  values=g.values(f)[0].copy()
  # At exactly the source parameter state, RESET and KEEP are identical.
  # No optimizer momentum, mutable BN stats, or history clearing is involved.
  if f[3]==0.:values[1]=0.
  return int(np.r_[0.,values].argmax())
 return choose
results=[]
for sid in split['evaluation']:
 x,y=load_subject(sid,kind);hf=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'];preds={}
 for typ,g in gates.items():
  name='identity_'+typ;preds[name],_,tm=stream(m,x,'controller',lr,identity_gate(g),hf)
  results.append({'subject':sid,'method':name,**metrics(y,preds[name]),**tm})
 np.savez_compressed(out/f'identity_predictions_s{sid:02d}.npz',y=y,**preds);print('IDENTITY',sid,{k:metrics(y,v)['macro_f1'] for k,v in preds.items()},flush=True)
(out/'identity_control.json').write_text(json.dumps({'results':results,'rule':'If pre-update parameter drift is exactly zero, analytically set predicted RESET advantage to zero: RESET=KEEP under the registered parameter-only reset semantics. Same rule and same source replay rows applied to future/current/confidence controllers. No target-label selection.'},indent=2));print('IDENTITY CONTROL COMPLETE',flush=True)

from adaptation import *
from controllers import fit_gate
import argparse
from sklearn.metrics import roc_auc_score
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent;m,ck=load_model(path);split=CFG['folds'][ck['fold']];kind=ck['kind'];lr=json.loads((out/'selection.json').read_text())['lr']
gate=fit_gate(json.loads((out/'controller_diagnostics.json').read_text())+json.loads((out/'source_replay_diagnostics.json').read_text()))
allrows=[];summ=[]
for sid in split['evaluation']:
 x,y=load_subject(sid,kind);hf=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'];pr,ds,tm=stream(m,x,'controller',lr,gate,hf,True,sid)
 saved=np.load(out/f'supplement_predictions_s{sid:02d}.npz')['refined_controller'];assert np.allclose(pr,saved,atol=1e-6,rtol=1e-5),'Diagnostic trajectory differs from evaluated refined policy'
 rows=score_diagnostics(ds,y,sid,'refined_controller')
 for r in rows:r['action']=int(gate(r['features']))
 allrows.extend(rows);r4=[r for r in rows if r['h']==4];gain=np.array([r['write_gain'] for r in r4]);act=np.array([r['action'] for r in r4]);entdrop=np.array([r['features'][5]<r['features'][0] for r in r4]);current=np.array([r['current_write_gain'] if r['current_write_gain'] is not None else np.nan for r in r4]);good=current>0
 summary={'subject':sid,'sampled_opportunities':len(r4),'sampled_commits':int((act==1).sum()),'sampled_resets':int((act==2).sum()),'harmful_commits_fraction':float((gain[act==1]<-1e-6).mean()) if (act==1).any() else None,'commit_mean_gain':float(gain[act==1].mean()) if (act==1).any() else None,'entropy_decrease_future_harm_fraction':float((gain[entdrop]<-1e-6).mean()) if entdrop.any() else None,'current_task_improvement_future_harm_fraction':float((gain[good]<-1e-6).mean()) if good.any() else None,'prediction_replay_match':True}
 q=gate.values(np.array([r['features'] for r in r4]))[:,0];labels=gain>0;summary['heldout_write_auc']=float(roc_auc_score(labels,q)) if len(set(labels))>1 else None
 summ.append(summary);print(json.dumps(summary),flush=True)
(out/'policy_diagnostics.json').write_text(json.dumps({'summary':summ,'rows':allrows}));print('POLICY DIAGNOSTICS COMPLETE',flush=True)

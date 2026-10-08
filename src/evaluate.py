from adaptation import *
import argparse,pickle,warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore',message='y_pred contains classes not in y_true')
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
path=Path(args.model);out=path.parent;m,ck=load_model(path);kind=ck['kind'];split=CFG['folds'][ck['fold']]
seed_all(ck['seed']);start=time.perf_counter()
def log(*a):print(*a,flush=True)
def subj(sid):
 x,y=load_subject(sid,kind);hf=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')['features'];return x,y,hf
# Hyperparameter selection uses base validation only, never controller/evaluation labels.
if (out/'selection.json').exists():sel=json.loads((out/'selection.json').read_text())
else:
 scores=[]
 for lr in CFG['adaptation']['lr_candidates']:
  per=[]
  for sid in split['base_val']:
   x,y,hf=subj(sid);pr,_,tm=stream(m,x,'continuous',lr,history_features=hf);per.append(metrics(y,pr)['macro_f1'])
  scores.append({'lr':lr,'macro_f1':float(np.mean(per))});log('LR',lr,per)
 lr=max(scores,key=lambda r:r['macro_f1'])['lr']
 sel={'lr':lr,'source_validation':scores,'entropy_threshold':.5,'period':20,'reset_period':100,'projection_relative_radius':.1,'selection_subjects':split['base_val']}
 (out/'selection.json').write_text(json.dumps(sel,indent=2))
lr=sel['lr'];log('SELECTED',sel)
# Mixed state distribution: sustained and periodic update trajectories, same permitted state transitions.
if (out/'controller_diagnostics.json').exists():train_rows=json.loads((out/'controller_diagnostics.json').read_text())
else:
 train_rows=[]
 for sid in split['controller_train']:
  x,y,hf=subj(sid)
  for mode in ['continuous','periodic']:
   _,ds,tm=stream(m,x,mode,lr,history_features=hf,collect=True,subject=sid)
   train_rows.extend(score_diagnostics(ds,y,sid,mode));log('CONTROLLER_DATA',sid,mode,len(ds),tm['seconds'])
 (out/'controller_diagnostics.json').write_text(json.dumps(train_rows))

class Gate:
 def __init__(self,reg,indices):self.reg=reg;self.indices=indices
 def values(self,f):return self.reg.predict(np.atleast_2d(f)[:,self.indices])
 def __call__(self,f):return np.r_[0.,self.values(f)[0]].argmax()
def fit_gate(rows,typ):
 rows=[r for r in rows if r['h']==4 and (typ!='current' or r['current_write_gain'] is not None)]
 xx=np.array([r['features'] for r in rows]);prefix='current_' if typ=='current' else ''
 yy=np.array([[r[prefix+'write_gain'],r[prefix+'reset_gain']] for r in rows])
 indices=[0,4] if typ=='confidence' else list(range(xx.shape[1]))
 reg=make_pipeline(StandardScaler(),Ridge(alpha=100.))
 # Equal total weight per source subject, to avoid long nights dominating controller fit.
 ss=np.array([r['subject'] for r in rows]);weights=np.array([len(ss)/(len(set(ss))*sum(ss==s)) for s in ss])
 reg.fit(xx[:,indices],yy,ridge__sample_weight=weights)
 return Gate(reg,indices)
gates={t:fit_gate(train_rows,t) for t in ['future','confidence','current']}
# Leave-one-controller-subject-out observation diagnostic, no target leakage.
cv=[]
for sid in split['controller_train']:
 train=[r for r in train_rows if r['subject']!=sid];test=[r for r in train_rows if r['subject']==sid and r['h']==4]
 gains=np.array([[0.,r['write_gain'],r['reset_gain']] for r in test]);features=np.array([r['features'] for r in test]);row={'subject':sid,'n':len(test),'oracle_ce_gain':float(gains.max(1).mean()),'always_write_gain':float(gains[:,1].mean())}
 for typ in gates:
  g=fit_gate(train,typ);q=g.values(features);acts=np.c_[np.zeros(len(q)),q].argmax(1)
  row[typ+'_gain']=float(gains[np.arange(len(gains)),acts].mean());row[typ+'_commit_rate']=float((acts==1).mean())
  labels=gains[:,1]>0
  row[typ+'_write_auc']=float(roc_auc_score(labels,q[:,0])) if len(set(labels))>1 else None
 cv.append(row)
log('CONTROLLER_LOSO',cv);(out/'controller_cv.json').write_text(json.dumps(cv,indent=2))
with (out/'controllers.pkl').open('wb') as f:pickle.dump(gates,f)
results=[];eval_rows=[]
for sid in split['evaluation']:
 x,y,hf=subj(sid);records={};preds={};t=time.perf_counter();frozen=predict(m,x);preds['frozen']=frozen;records['frozen']={'seconds':time.perf_counter()-t}
 preds['smooth']=smooth(frozen)
 for name,fn in [('t3a',t3a),('causal_bn',causal_bn)]:
  t=time.perf_counter();preds[name]=fn(m,x);records[name]={'seconds':time.perf_counter()-t}
 modes={'continuous':('continuous',lr,None),'tiny_lr':('continuous',min(CFG['adaptation']['lr_candidates']),None),'periodic':('periodic',lr,None),'low_entropy':('low_entropy',lr,None),'reset_periodic':('reset_periodic',lr,None),**{k+'_controller':('controller',lr,v) for k,v in gates.items()}}
 for name,(mode,rate,gate) in modes.items():
  collect=name=='continuous';pr,ds,tm=stream(m,x,mode,rate,gate,hf,collect,sid)
  preds[name]=pr;records[name]=tm
  if collect:eval_rows.extend(score_diagnostics(ds,y,sid,mode))
  log('EVAL',sid,name,metrics(y,pr)['macro_f1'],tm['seconds'])
 np.savez_compressed(out/f'predictions_s{sid:02d}.npz',y=y,**preds)
 for name,pr in preds.items():
  row={'subject':sid,'method':name,**metrics(y,pr),**records.get(name,{})};results.append(row)
 (out/'evaluation.json').write_text(json.dumps(results,indent=2));(out/'evaluation_diagnostics.json').write_text(json.dumps(eval_rows))
summary={}
for method in sorted(set(r['method'] for r in results)):
 rr=[r for r in results if r['method']==method];ff=[r for r in results if r['method']=='frozen'];delta=np.array([a['macro_f1']-b['macro_f1'] for a,b in zip(rr,ff)])
 summary[method]={'macro_f1':float(np.mean([r['macro_f1'] for r in rr])),'delta_f1':float(delta.mean()),'neg_rate_001':float((delta<-.01).mean()),'per_subject_delta':delta.tolist()}
(out/'summary.json').write_text(json.dumps({'summary':summary,'elapsed':time.perf_counter()-start,'model':str(path),'cpu_ids':sorted(os.sched_getaffinity(0))},indent=2));log('COMPLETE',summary)

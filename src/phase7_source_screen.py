from common import *
from adaptation import stream,score_diagnostics
from controllers import fit_gate
from sklearn.ensemble import HistGradientBoostingRegressor
import time,itertools
OUT=ROOT/"phase7";OUT.mkdir(exist_ok=True)
protocol=dict(purpose="Source-only screen: controller-specific high learning rate and nonlinear gain estimator; no new-cohort labels used",models=["context11_young_17","context11_mixed_17"],lr=.01,controller_train=[7,17,27,37],validation=[5,15,25,35],regressor="HGB100iterations7leaves,minleaf30,l2=10,lr=.05,early_stopping=False; source-normalized targets; compare Ridge",interpretation="LR chosen for continuous adaptation may suppress exploitable WRITE gains; test a fixed larger candidate step with source-held-out controller validation",gpu_allowed=False,cpu_ids=list(range(8,24)))
(OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
def subj(s):
 root=ROOT/("data/processed" if s<20 else "phase2/processed");d=np.load(root/f"s{s:02d}_meta.npz");hf=d["features"];pad=np.pad(hf,((10,0),(0,0)),mode="edge");return np.concatenate([pad[j:j+len(hf)] for j in range(11)],axis=1),d["y"],hf
class NonlinearGate:
 def __init__(self,rows,typ):
  rows=[r for r in rows if r["h"]==4 and (typ!="current" or r["current_write_gain"] is not None)];self.indices=[0,4] if typ=="confidence" else list(range(13));x=np.array([r["features"] for r in rows])[:,self.indices];prefix="current_" if typ=="current" else "";y=np.array([[r[prefix+"write_gain"],r[prefix+"reset_gain"]] for r in rows]);self.mean=y.mean(0);self.scale=np.maximum(y.std(0),1e-6);ss=np.array([r["subject"] for r in rows]);weights=np.array([len(ss)/(len(set(ss))*np.sum(ss==s)) for s in ss]);self.models=[]
  for j in range(2):
   reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=17);reg.fit(x,(y[:,j]-self.mean[j])/self.scale[j],sample_weight=weights);self.models.append(reg)
 def values(self,f):
  x=np.atleast_2d(f)[:,self.indices];return np.column_stack([reg.predict(x) for reg in self.models])*self.scale+self.mean
for name in protocol["models"]:
 out=OUT/name;out.mkdir(exist_ok=True)
 if (out/"validation.json").exists():continue
 ck=torch.load(ROOT/"phase4/results"/name/"model.pt",map_location="cpu",weights_only=False);m=SpectralNet(ck["dim"]);m.load_state_dict(ck["state"]);m.eval();data={s:subj(s) for s in protocol["controller_train"]+protocol["validation"]};path=out/"source_rows.json"
 if not path.exists():
  rows=[]
  for s in protocol["controller_train"]:
   x,y,hf=data[s]
   for mode in ["continuous","periodic"]:
    _,ds,_=stream(m,x,mode,protocol["lr"],history_features=hf,collect=True,subject=s);rows.extend(score_diagnostics(ds,y,s,mode))
  path.write_text(json.dumps(rows));print("HIGH LR DATA",name,flush=True)
 rows=json.loads(path.read_text());cv=[]
 for s in protocol["controller_train"]:
  train=[v for v in rows if v["subject"]!=s];test=[v for v in rows if v["subject"]==s and v["h"]==4];features=np.array([v["features"] for v in test]);gains=np.array([[0.,v["write_gain"],v["reset_gain"]] for v in test])
  for learner,typ in itertools.product(["ridge","hgb"],["future","confidence","current"]):
   gate=fit_gate(train,typ) if learner=="ridge" else NonlinearGate(train,typ);pred=gate.values(features);pred[features[:,3]==0,1]=0;actions=np.column_stack([np.zeros(len(pred)),pred]).argmax(1);cv.append(dict(subject=s,learner=learner,target=typ,realized_gain=float(gains[np.arange(len(actions)),actions].mean()),write_or_keep_oracle=float(gains[:,:2].max(1).mean()),all_action_oracle=float(gains.max(1).mean()),action_counts=np.bincount(actions,minlength=3).tolist()))
 (out/"source_loso.json").write_text(json.dumps(cv,indent=2));results=[]
 for learner,typ in itertools.product(["ridge","hgb"],["future","confidence","current"]):
  gate=fit_gate(rows,typ) if learner=="ridge" else NonlinearGate(rows,typ)
  for s in protocol["validation"]:
   saved=out/f"{learner}_{typ}_s{s:02d}.json"
   if saved.exists():results.append(json.loads(saved.read_text()));continue
   x,y,hf=data[s]
   def choose(f):
    value=gate.values(f)[0].copy()
    if f[3]==0.:value[1]=0.
    return int(np.r_[0.,value].argmax())
   pr,_,tm=stream(m,x,"controller",protocol["lr"],choose,hf);row=dict(subject=s,learner=learner,target=typ,**metrics(y,pr),timing=tm);saved.write_text(json.dumps(row,indent=2));results.append(row);print("SOURCE VALIDATION",name,learner,typ,s,row["macro_f1"],flush=True)
 frozen=[dict(subject=s,**metrics(data[s][1],predict(m,data[s][0]))) for s in protocol["validation"]];(out/"validation.json").write_text(json.dumps(dict(results=results,frozen=frozen,source_only=True),indent=2));print("SOURCE SCREEN COMPLETE",name,flush=True)

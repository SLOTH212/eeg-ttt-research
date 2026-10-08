from phase37_screen import load as load_carrier
from phase30_masked_graph import *
from phase2_temporal import ema
import hashlib,datetime
OUT=ROOT/"phase40"
def lock():
 selection=json.loads((ROOT/"phase39/source_selection.json").read_text());assert selection["complete"];assert json.loads((ROOT/"phase39/audit.json").read_text())["status"]=="PASS";assert json.loads((ROOT/"phase35/audit.json").read_text())["classifier_state_exact"];protocol=json.loads((OUT/"protocol.json").read_text());checkpoints={}
 for variant,folder in [("fitted","phase35/mixed"),("unfitted","phase28/mixed_joint_0")]:
  file=ROOT/folder/"model.pt";checkpoints[variant]=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 token_hashes={str(sid):hashlib.sha256((ROOT/"phase31/tokens"/f"s{sid:02d}.npz").read_bytes()).hexdigest() for sid in protocol["subjects"]}
 data=dict(source_selection=selection,source_selection_sha256=hashlib.sha256((ROOT/"phase39/source_selection.json").read_bytes()).hexdigest(),checkpoints=checkpoints,token_cache_sha256=token_hashes,locked_before_target_inference=True)
 path=OUT/"selection_lock.json"
 if path.exists():assert json.loads(path.read_text())==data
 else:path.write_text(json.dumps(data,indent=2))
 return data,protocol
def run():
 locked,protocol=lock();choices={(d["variant"],d["objective"]):d["selected_lr"] for d in locked["source_selection"]["decisions"]};settings={"fitted_selected":("fitted","masked_token",choices["fitted","masked_token"]),"unfitted_selected":("unfitted","masked_token",choices["unfitted","masked_token"]),"unfitted_matched":("unfitted","masked_token",choices["fitted","masked_token"]),"entropy_selected":("unfitted","entropy",choices["unfitted","entropy"])}
 models={v:load_carrier("mixed",v)[0] for v in ["fitted","unfitted"]};graphs={};immutable={}
 for v,m in models.items():
  writable={n for n,p in m.named_parameters() if p.requires_grad};immutable[v]={n:z.detach().clone() for n,z in m.state_dict().items() if n not in writable}
 for name,(variant,objective,lr) in settings.items():
  if lr>0:graphs[name]=MaskedGraphAdapter(models[variant],lr,torch.zeros(1,2,2,15,200,device="cuda")) if objective=="masked_token" else GraphAdapter(models[variant],"entropy",lr,torch.zeros(1,2,2,15,200,device="cuda"))
 states={v:torch.load(locked["checkpoints"][v]["path"],map_location="cpu",weights_only=False)["state"] for v in models};results=[]
 for sid in protocol["subjects"]:
  file=OUT/f"s{sid:02d}.npz";done=file.with_suffix(".json")
  if done.exists():results.append(json.loads(done.read_text()));continue
  start=time.perf_counter();x,y=prepared(sid);models["fitted"].load_state_dict(states["fitted"]);base=probabilities(models["fitted"],x);predictions=dict(frozen=base,weak_ema=ema(base,.8),strong_ema=ema(base,.5));cache=ROOT/"phase31/tokens"/f"s{sid:02d}.npz";assert hashlib.sha256(cache.read_bytes()).hexdigest()==locked["token_cache_sha256"][str(sid)];tokens=np.load(cache)["tokens"];assert tokens.shape==(len(x),2,30) and tokens.min()>=0 and tokens.max()<8192
  for name,(variant,objective,lr) in settings.items():
   if lr==0:predictions[name]=base.copy()
   else:
    graph=graphs[name];graph.reset();pred=[]
    for t in range(len(x)):
     argument=torch.tensor(tokens[t:t+1].astype(np.int64),device="cuda") if objective=="masked_token" else torch.ones(5,device="cuda")/5;q=graph.step(x[t:t+1].cuda(),argument);pred.append(q.detach().cpu().numpy()[0].copy());assert bool(graph.finite)
    predictions[name]=np.array(pred);assert abs(predictions[name][0]-base[0]).max()<2e-5;assert all(float((p-v).detach().norm())<=radius+1e-5 for p,v,radius in zip(graph.params,graph.initial,graph.radii))
   predictions[name+"_ema08"]=ema(predictions[name],.8)
  assert all(torch.equal(m.state_dict()[n],v) for variant,m in models.items() for n,v in immutable[variant].items());assert set(predictions)==set(protocol["methods"]);np.savez_compressed(file,y=y,**predictions);mm={name:metrics(y,p) for name,p in predictions.items()};rec=dict(subject=sid,cohort="older" if sid>=40 else "middle" if sid>=20 else "young",methods=mm,nonwritable_unchanged=True,radius_verified=True,source_selection_locked=True,pre_update_predictions=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(rec,indent=2));results.append(rec);print("HEAD ONLY TRANSFER",sid,{k:round(v["macro_f1"]*100,3) for k,v in mm.items()},flush=True)
 (OUT/"complete.json").write_text(json.dumps(dict(complete=True,source="mixed",people=len(results),methods=len(protocol["methods"]),exploratory=True,settings=settings,results=results),indent=2))
if __name__=="__main__":run()

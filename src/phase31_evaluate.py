from phase30_masked_graph import *
from phase26_reconstruction import cls_bias
from phase2_temporal import ema
import hashlib
OUT31=ROOT/"phase31"
def lock_selection():
 path=OUT31/"selection_lock.json";selection=json.loads((ROOT/"phase30/source_selection.json").read_text());assert selection["complete"];assert len(list((ROOT/"phase30").glob("young_w*/complete.json")))==16;assert json.loads((ROOT/"phase30/audit.json").read_text())["complete"];assert json.loads((ROOT/"phase30/frozen_prefix_audit.json").read_text())["status"]=="PASS";checkpoints={}
 for weight in [0.,.01]:
  file=ROOT/"phase28"/f"young_joint_{weight:g}"/"model.pt";checkpoints[str(weight)]=dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 data=dict(source_selection=selection,checkpoints=checkpoints,source_selection_sha256=hashlib.sha256((ROOT/"phase30/source_selection.json").read_bytes()).hexdigest(),primary_weight=.01,locked_before_target_inference=True)
 if path.exists():
  old=json.loads(path.read_text());assert old==data;return old
 path.write_text(json.dumps(data,indent=2));return data
def run(weight):
 lock=lock_selection();out=OUT31/f"young_joint_{weight:g}";out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 choices={d["objective"]:d for d in lock["source_selection"]["decisions"] if d["weight"]==weight};assert len(choices)==2;ck=torch.load(lock["checkpoints"][str(weight)]["path"],map_location="cpu",weights_only=False);model=load_model(weight);graphs={}
 for objective in ["entropy","masked_token"]:
  lr=choices[objective]["selected_lr"]
  if lr>0:
   example=torch.zeros(1,2,2,15,200,device="cuda");graphs[objective]=MaskedGraphAdapter(model,lr,example) if objective=="masked_token" else GraphAdapter(model,"entropy",lr,example)
 frozen={n:v.detach().clone() for n,v in model.state_dict().items() if not dict(model.named_parameters()).get(n,torch.tensor(0.)).requires_grad};teacher=None
 if "masked_token" in graphs:
  teacher=vqnsp.vqnsp_encoder_base_decoder_3x200x12(code_dim=64).eval();teacher.load_state_dict(torch.load(ROOT/"phase14/tokenizer_weights_only.pt",map_location="cpu",weights_only=True),strict=True);teacher.requires_grad_(False);midpoint_positions(teacher.encoder);teacher.cuda();teacherstate={k:v.clone() for k,v in teacher.state_dict().items()}
 protocol=json.loads((OUT31/"protocol.json").read_text());results=[]
 for sid in protocol["subjects"]:
  dest=out/f"s{sid:02d}.npz";done=dest.with_suffix(".json")
  if done.exists():results.append(json.loads(done.read_text()));continue
  start=time.perf_counter();x,y=prepared(sid);model.load_state_dict(ck["state"]);base=probabilities(model,x);predictions=dict(frozen=base,weak_ema=ema(base,.8),strong_ema=ema(base,.5));tokens=None
  if teacher is not None:
   cache=OUT31/"tokens"/f"s{sid:02d}.npz"
   if cache.exists():tokens=np.load(cache)["tokens"]
   else:
    parts=[]
    with cls_bias(teacher.encoder,8):
     for b in x.split(32):parts.append(tokenizer_indices(teacher,b.flatten(0,1).cuda()).cpu().numpy().reshape(-1,2,30).astype(np.uint16))
    tokens=np.concatenate(parts);np.savez_compressed(cache,tokens=tokens);assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in teacherstate.items())
  for objective,name in [("entropy","entropy_selected"),("masked_token","masked_selected")]:
   if objective not in graphs:predictions[name]=base.copy()
   else:
    graph=graphs[objective];graph.reset();pred=[]
    for t in range(len(y)):
     xx=x[t:t+1].cuda();argument=torch.tensor(tokens[t:t+1].astype(np.int64),device="cuda") if objective=="masked_token" else torch.ones(5,device="cuda")/5
     pred.append(graph.step(xx,argument).detach().cpu().numpy()[0].copy());assert bool(graph.finite)
    predictions[name]=np.array(pred);assert abs(predictions[name][0]-base[0]).max()<2e-5;assert all(float((p-v).norm())<=radius+1e-5 for p,v,radius in zip(graph.params,graph.initial,graph.radii))
   predictions[name+"_ema08"]=ema(predictions[name],.8)
  assert all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items());np.savez_compressed(dest,y=y,**predictions);metrics_by_method={name:metrics(y,p) for name,p in predictions.items()};record=dict(source="young",weight=weight,subject=sid,cohort="older" if sid>=40 else "middle" if sid>=20 else "young",methods=metrics_by_method,nonwritable_unchanged=True,radius_verified=True,teacher_unchanged=True,source_selection_locked=True,seconds=time.perf_counter()-start);done.write_text(json.dumps(record,indent=2));results.append(record);print("JOINT TRANSFER",weight,sid,{name:round(m["macro_f1"]*100,3) for name,m in metrics_by_method.items()},flush=True)
 (out/"complete.json").write_text(json.dumps(dict(weight=weight,results=results,exploratory=True,source_selection=choices),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--weight",type=float,required=True);a=parser.parse_args();run(a.weight)

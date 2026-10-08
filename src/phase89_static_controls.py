from phase86_meta_waveform import *
from phase2_temporal import ema
import pandas as pd
O=ROOT/"phase89";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def features(m,x):
 return (m.encoder.encoder(m.encoder.patch_embedding(x)).mean(2).flatten(1)-m.mean)/m.scale
def run():
 assert json.loads((ROOT/"phase88/audit.json").read_text())["status"]=="PASS";pr86=lock()
 pr=dict(purpose="Source-only full-record static alternatives after exploratory meta-write diagnostics; no target rollout.",subjects=pr86["controllers"]+pr86["held_source_validation"],primary_people=pr86["held_source_validation"],methods=["frozen","source_mean_current","source_mean_future","static_head_current","static_head_future"],ema=[1.,.8,.5],head_training="Only existing 400->5 classifier linear head,2005 parameters. Exact317 meta pulses, current317/future1268 labels,5epochs,batch4pulses,AdamWlr.001wd0,clip1,equal-personweights,dedicated order17; no selection.",static_mean="phase88 mean source corrections applied once before target-independent inference, no per-window updates.",comparability="Same labels and label presentations as meta branches; parameter roles/counts and compute differ. Meta-head training uses40200parameters, static classifier head2005. All source people previously observed.",meta_protocol_sha256=hashlib.sha256((OUT86/"protocol.json").read_bytes()).hexdigest(),directions_sha256={w:hashlib.sha256((ROOT/"phase88"/(w+"_source_directions.npz")).read_bytes()).hexdigest() for w in ["current","future"]})
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest())
 m=meta_model();named=ln_named(m);names=[n for n,p in named];initial={n:p.detach().clone() for n,p in named};m.requires_grad_(False);state=snapshot(m);cached={};records=[];counts={}
 for sid in pr86["meta_subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];counts[sid]=len(times)
  for t0 in times:
   t=int(t0);x,y=sparse(sid,t)
   with torch.no_grad():z=features(m,x);err=float(abs(m.head(z)-m(x)).max());assert err==0
   cached[sid,t]=(z.detach(),y);records.append((sid,t))
 print("FEATURES",len(records),flush=True);trained={};training={}
 for which in ["current","future"]:
  head=nn.Linear(400,5).cuda();head.load_state_dict(m.head.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0);gen=torch.Generator(device="cpu").manual_seed(17);orders=[];history=[];labelset=set()
  for sid,t in records:labelset.update((sid,t+j) for j in ([0] if which=="current" else [1,2,3,4]))
  for epoch in range(5):
   order=torch.randperm(len(records),generator=gen).tolist();orders.append(order);losses=[]
   for i in range(0,len(order),4):
    batch=order[i:i+4];opt.zero_grad();loss=torch.zeros((),device="cuda")
    for idx in batch:
     sid,t=records[idx];z,y=cached[sid,t];z,y=(z[:1],y[:1]) if which=="current" else (z[1:],y[1:]);weight=len(records)/(len(counts)*counts[sid]*len(batch));loss=loss+weight*nn.functional.cross_entropy(head(z),y)
    loss.backward();gn=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
   history.append(dict(epoch=epoch+1,task_ce=float(np.mean(losses))))
  assert orders==json.loads((OUT86/which/"training_orders.json").read_text())["orders"];assert len(labelset)==(317 if which=="current" else 1268);trained[which]={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};torch.save(trained[which],O/(which+"_classifier_head.pt"));training[which]=dict(labels_used=[list(v) for v in sorted(labelset)],label_count=len(labelset),label_presentations=len(labelset)*5,epochs=5,optimizer_steps=5*((len(records)+3)//4),parameters=sum(p.numel() for p in head.parameters()),history=history,training_order_matches_meta_exact=True);print("STATIC HEAD",which,training[which]["label_count"],history[-1],flush=True)
 assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());(O/"training_audit.json").write_text(json.dumps(dict(status="PASS",records=training,classifier_encoder_frozen=True),indent=2))
 checks=[]
 for sid in pr["subjects"]:
  x,y=prepared(sid);predictions={};m.load_state_dict(state)
  # Store baseline features only in memory; fitted heads require no encoder rerun.
  zs=[];ps=[]
  with torch.no_grad():
   for batch in x.split(128):
    z=features(m,batch.cuda());zs.append(z.cpu());ps.append(m.head(z).softmax(1).cpu().numpy())
  z=torch.cat(zs);predictions["frozen"]=np.concatenate(ps)
  for which in ["current","future"]:
   m.load_state_dict(state);m.head.load_state_dict(trained[which])
   with torch.no_grad():predictions["static_head_"+which]=np.concatenate([m.head(b.cuda()).softmax(1).cpu().numpy() for b in z.split(128)])
   m.load_state_dict(state);vector=torch.tensor(np.load(ROOT/"phase88"/(which+"_source_directions.npz"))["mean"],device="cuda",dtype=torch.float32);offset=0
   with torch.no_grad():
    for name,p in m.named_parameters():
     if name in initial:n=p.numel();p.copy_(initial[name]+vector[offset:offset+n].view_as(p));offset+=n
   assert offset==9600;predictions["source_mean_"+which]=probabilities(m,x)
   changed=[n for n,v in m.state_dict().items() if not torch.equal(v.cpu(),state[n])];assert all(n in initial for n in changed)
  m.load_state_dict(state);assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());p0=predictions["frozen"]
  with torch.no_grad():one=m(x[:1].cuda()).softmax(1).cpu().numpy()
  err=float(abs(p0[:1]-one).max());assert err<2e-5
  if sid in pr["primary_people"]:
   old=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")["cbramod_seed17"];olderr=float(abs(p0-old).max());assert olderr<2e-5 and np.array_equal(p0.argmax(1),old.argmax(1))
  else:olderr=None
  for p in predictions.values():assert p.shape==(len(y),5) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<2e-6
  np.savez_compressed(O/"predictions"/f"s{sid:02d}.npz",**predictions);checks.append(dict(subject=sid,windows=len(y),batch_one_error=err,prior_frozen_probability_error=olderr,state_restored_exact=True));print("FULL SOURCE",sid,len(y),flush=True)
 (O/"execution_audit.json").write_text(json.dumps(dict(status="PASS",checks=checks,source_only=True,no_online_updates=True),indent=2));(O/"complete.json").write_text(json.dumps(dict(people=8,methods=5,streams=120)))
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()

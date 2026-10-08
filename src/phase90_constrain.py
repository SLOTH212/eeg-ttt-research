from phase86_meta_waveform import *
O=ROOT/"phase90";O.mkdir(exist_ok=True);(O/"source_cases").mkdir(exist_ok=True)
def run():
 pr86=lock()
 protocol=dict(purpose="Source-only analytic reconstruction-constrained projection of task-meta auxiliary heads; exploratory new branch, phase87failure preserved.",source_subjects=pr86["meta_subjects"],source_pulses="Same317times, same73000+sid*100000+t masks; no additional classification supervision.",heads=["future","current"],interpolation="H(beta)=H_initial+beta*(H_meta-H_initial), single shared beta per head in[0,1].",constraint="For EACH of5source people preserve at least half the initial mean-window reconstruction advantage over zero and over shuffled-input reconstruction. Both initial advantages must be strictly positive; otherwise no constrained-head candidate.",shuffle="Within source person, donor pulse circular offset floor(number_pulses/2); recipient mask; offline source fitting only.",solution="Exact quadratic coefficients in beta from linear head outputs. Largest beta in feasible interval connected to0, capped1, multiplied by(1-1e-6) for numeric slack. No classification metric used, no grid search.",validation="Source4 separate reconstruction and actual-write checks only after rule fixed; all source people previously observed in project.",label_cost_inherited={"future":1268,"current":317},extra_labels=0,parent_protocol_sha256=hashlib.sha256((OUT86/"protocol.json").read_bytes()).hexdigest())
 (O/"protocol.json").write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest());m=meta_model();m.requires_grad_(False);state=snapshot(m);initial={k:v.detach().cpu().double().numpy() for k,v in m.encoder.proj_out[0].state_dict().items()};heads={w:{k:v.double().numpy() for k,v in torch.load(OUT86/w/"head.pt",map_location="cpu",weights_only=False)["head"].items()} for w in protocol["heads"]}
 stats=[]
 for sid in protocol["source_subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");zs=[];ss=[];targets=[];counts=[];person={w:[] for w in heads}
  for i,t0 in enumerate(times):
   t=int(t0);donor=int(times[(i+len(times)//2)%len(times)]);x=torch.tensor(prep(np.array(raw[t:t+1]),[.1,45]),device="cuda");other=torch.tensor(prep(np.array(raw[donor:donor+1]),[.1,45]),device="cuda");mask=mask_for(sid,t)
   with torch.no_grad():z=m.encoder.encoder(m.encoder.patch_embedding(x,mask))[mask].cpu().numpy();sh=m.encoder.encoder(m.encoder.patch_embedding(other,mask))[mask].cpu().numpy()
   target=x[mask].cpu().numpy();zs.append(z);ss.append(sh);targets.append(target);counts.append(len(z));zf=z.astype(float);sf=sh.astype(float);yf=target.astype(float);p0=zf@initial["weight"].T+initial["bias"];s0=sf@initial["weight"].T+initial["bias"];error=p0-yf;se=s0-yf
   for which,h in heads.items():
    delta=zf@(h["weight"]-initial["weight"]).T+(h["bias"]-initial["bias"]);sd=sf@(h["weight"]-initial["weight"]).T+(h["bias"]-initial["bias"])
    person[which].append([np.mean(delta**2),2*np.mean(error*delta),np.mean(error**2),np.mean(sd**2),2*np.mean(se*sd),np.mean(se**2),np.mean(yf**2)])
  np.savez_compressed(O/"source_cases"/f"s{sid:02d}.npz",times=times,counts=np.array(counts),z=np.concatenate(zs),shuffle_z=np.concatenate(ss),target=np.concatenate(targets))
  for which,values in person.items():
   a,b,c,sa,sb,sc,zero=np.mean(values,axis=0);stats.append(dict(subject=sid,head=which,a=a,b=b,c=c,shuffle_a=sa,shuffle_b=sb,shuffle_c=sc,zero_mse=zero));print("SOURCE CONSTRAINT",sid,which,"initial",c,"zero",zero,"shuffle",sc,flush=True)
 (O/"quadratic_coefficients.json").write_text(json.dumps(stats,indent=2));solutions={}
 def boundary(a,b,c):
  assert c<0
  roots=np.roots([a,b,c] if abs(a)>1e-20 else [b,c]) if abs(a)>1e-20 or abs(b)>1e-20 else []
  positives=[float(v.real) for v in roots if abs(v.imag)<1e-10 and v.real>0]
  return min([1.]+positives)
 for which,h in heads.items():
  rows=[r for r in stats if r["head"]==which];feasible=all(r["c"]<min(r["zero_mse"],r["shuffle_c"]) for r in rows)
  if not feasible:solutions[which]=dict(feasible=False,reason="Initial head does not beat both controls for every source person; no rule relaxation.");continue
  limits=[]
  for r in rows:
   limits.append(boundary(r["a"],r["b"],.5*(r["c"]-r["zero_mse"])))
   limits.append(boundary(r["a"]-r["shuffle_a"],r["b"]-r["shuffle_b"],.5*(r["c"]-r["shuffle_c"])))
  beta=min(limits)*(1-1e-6);head={k:torch.tensor(initial[k]+beta*(h[k]-initial[k]),dtype=torch.float32) for k in initial};torch.save(dict(head=head,beta=beta,source_only=True),O/(which+"_constrained_head.pt"));solutions[which]=dict(feasible=True,beta=beta,individual_constraint_boundaries=limits,head_sha256=hashlib.sha256((O/(which+"_constrained_head.pt")).read_bytes()).hexdigest())
 assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());(O/"solutions.json").write_text(json.dumps(solutions,indent=2));(O/"complete.json").write_text(json.dumps(dict(status="PASS",source_people=5,source_pulses=317,extra_classification_labels=0,classifier_unchanged_exact=True)))
 print("SOLUTIONS",solutions,flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()

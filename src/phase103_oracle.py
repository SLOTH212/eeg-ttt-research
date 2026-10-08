from phase86_meta_waveform import *
import pandas as pd,datetime
O=ROOT/"phase103";O.mkdir(exist_ok=True)
def run():
 old=lock();protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Source-only supervised diagnostic controls for cold single WRITE: distinguish update direction and magnitude limitations, not deployable TTT.",controllers=old["controllers"],validation=old["held_source_validation"],times="Exact phase91 times; previously observed sourcepeople",variants=["oracle_current_clip","oracle_future_clip","oracle_current_matched","oracle_future_matched"],lr=.01,clip=1.,parameters="Same9600LN affine; initial fixed classifierseed17; no accumulatedhistory.",matched="Match per-pulse actual constrainedfuture-head update norm from phase91, without choosing on new outcomes.",labels="Current oracle uses current trueclass; future oracle uses allfour futuretrueclasses and is explicitly noncausal diagnostic. Neither is an unlabeled deployable method. No new people or targetlabels.",interpretation="Future-gradient control is a local optimizer positivecontrol, not global achievable bound. Do not add overlapping pulse outcomes to onlineMacroF1. No efficacygate or modelselection.",phase91_protocol_sha256=hashlib.sha256((ROOT/"phase91/protocol.json").read_bytes()).hexdigest(),references={f"s{s:02d}":hashlib.sha256((ROOT/"phase91"/f"s{s:02d}.json").read_bytes()).hexdigest() for s in old["controllers"]+old["held_source_validation"]})
 if (O/"protocol.json").exists():
  p=json.loads((O/"protocol.json").read_text());protocol["created_utc"]=p["created_utc"];assert p==protocol
 else:(O/"protocol.json").write_text(json.dumps(protocol,indent=2))
 m=meta_model();named=ln_named(m);params=[p for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 allrows=[]
 for group,sids in [("controllers",old["controllers"]),("validation",old["held_source_validation"])]:
  for sid in sids:
   dest=O/f"s{sid:02d}.json"
   if dest.exists():allrows.extend(json.loads(dest.read_text())["rows"]);continue
   previous=json.loads((ROOT/"phase91"/f"s{sid:02d}.json").read_text());refs={r["t"]:r for r in previous["rows"] if r["variant"]=="future"};rows=[]
   for t in previous["times"]:
    restore();x,y=sparse(sid,t);lp=m(x).double().log_softmax(1);before=-lp[torch.arange(5,device="cuda"),y];bp=lp.detach().exp().cpu().numpy();assert abs(bp[1:]-np.array(refs[t]["future_base_probabilities"])).max()<1e-10
    gradients={"current":torch.autograd.grad(before[0],params,retain_graph=True),"future":torch.autograd.grad(before[1:].mean(),params)}
    norms={k:torch.sqrt(sum(v.square().sum() for v in g)).detach() for k,g in gradients.items()};target=refs[t]["actual_update_norm"]
    for variant in protocol["variants"]:
     restore();which=variant.split("_")[1];g=gradients[which];norm=norms[which];assert torch.isfinite(norm) and norm>0
     scale=.01*torch.clamp(1/(norm+1e-6),max=1.) if variant.endswith("clip") else target/norm
     with torch.no_grad():
      for p,v,d in zip(params,initial,g):p.copy_(v-scale*d)
      actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=.01001
      if variant.endswith("matched"):assert abs(actual-target)<5e-6
      linear=float(-sum(((p-v).double()*f.double()).sum() for p,v,f in zip(params,initial,gradients["future"])))
      afterlp=m(x).double().log_softmax(1);wp=afterlp.exp().cpu().numpy();after=-afterlp[torch.arange(5,device="cuda"),y]
     yy=y[1:].cpu().numpy();b=bp[1:].argmax(1);w=wp[1:].argmax(1);gain=float((before.detach()[1:]-after[1:]).mean())
     rows.append(dict(group=group,subject=sid,t=t,variant=variant,actual_update_norm=actual,matched_reference_norm=target,gradient_norm=float(norm),current_ce_gain=float(before.detach()[0]-after[0]),future_ce_gain=gain,balanced_ce_gain=float(((before.detach()[1:]-after[1:])*cw[y[1:]]).mean()),linear_future_ce_gain=linear,linear_residual=gain-linear,future_labels=yy.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),corrected=int(((b!=yy)&(w==yy)).sum()),harmed=int(((b==yy)&(w!=yy)).sum()),argmax_changes=int((b!=w).sum())))
   restore();assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());dest.write_text(json.dumps(dict(subject=sid,group=group,rows=rows,state_restored_exact=True),indent=2));allrows.extend(rows);print("ORACLE DIAGNOSTIC",group,sid,len(rows),flush=True)
 df=pd.DataFrame(allrows);assert len(df)==2048;df.drop(columns=["future_labels","future_base_probabilities","future_write_probabilities"]).to_csv(O/"pulses.csv",index=False)
 (O/"audit.json").write_text(json.dumps(dict(status="PASS",pulse_rows=len(df),same_keep_probabilities_as_phase91=True,matched_norm_max_error=float(abs(df[df.variant.str.endswith("matched")].actual_update_norm-df[df.variant.str.endswith("matched")].matched_reference_norm).max()),model_state_restored_exact=True,source_oracle_only=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()

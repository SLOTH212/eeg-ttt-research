from phase52_meta_head import *
from phase29_forward_graph import LogitForward
OUT=ROOT/"phase52"
def load(variant):
 m=model_for("masked")
 if variant in ["future","current","reconstruction"]:m.load_state_dict(torch.load(OUT/variant/"model.pt",map_location="cpu",weights_only=False)["state"])
 initial=model_for("masked").cpu().state_dict()
 assert all(torch.equal(v.cpu(),initial[n]) for n,v in m.state_dict().items() if not n.startswith("recon_head."))
 return m
def validate():
 rows=[]
 for variant in ["initial","future","current","reconstruction"]:
  m=load(variant)
  for sid in [5,15,25,35]:
   z=np.load(ROOT/"phase46/cache"/f"s{sid:02d}.npz");features=torch.tensor(z["masked"],device="cuda");target=torch.tensor(z["target"],device="cuda");meanmse=float(np.square((z["target"]-m.recon_target_mean.cpu().numpy())/m.recon_target_scale.cpu().numpy()).mean())
   with torch.no_grad():
    xx=(features-m.recon_input_mean)/m.recon_input_scale;yy=(target-m.recon_target_mean)/m.recon_target_scale;pred=m.recon_head(xx);shuffle=m.recon_head(torch.roll(xx,len(xx)//2,0));mse=float((pred-yy).square().mean());sh=float((shuffle-yy).square().mean())
   rows.append(dict(variant=variant,subject=sid,mse=mse,source_mean_mse=meanmse,shuffled_mse=sh,validity_pass=mse<meanmse and mse<sh))
  del m
 pd.DataFrame(rows).to_csv(OUT/"reconstruction_validity.csv",index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
def run(variant):
 out=OUT/"evaluation"/variant;out.mkdir(parents=True,exist_ok=True)
 if (out/"complete.json").exists():return
 m=load("initial" if variant=="initial_normmatched" else variant);params=[p for p in m.parameters() if p.requires_grad];initial=[p.detach().clone() for p in params];state={n:v.clone() for n,v in m.state_dict().items()};forward=LogitForward(m,torch.zeros(5,2,2,15,200,device="cuda"));cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda");allrows=0
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in [7,17,27,37]:
  dest=out/f"s{sid:02d}.json"
  if dest.exists():allrows+=len(json.loads(dest.read_text())["rows"]);continue
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];rows=[];reference={(r["t"]):r for r in json.loads((ROOT/"phase47/masked"/f"s{sid:02d}.json").read_text())["rows"] if r["objective"]=="spectral" and r["lr"]==.01};matching={}
  if variant=="initial_normmatched":matching={r["t"]:r["proposed_update_norm"] for r in json.loads((OUT/"evaluation/future"/f"s{sid:02d}.json").read_text())["rows"]}
  for t0 in times:
   t=int(t0);restore();x,target,y=pulse(sid,t)
   with torch.no_grad():logits=forward.run(x).clone();bp=logits.softmax(1).cpu().numpy();before=-logits.log_softmax(1)[torch.arange(5,device="cuda"),y]
   aux=m.reconstruction_loss(x[:1],target[:1]);g=torch.autograd.grad(aux,params);norm=torch.sqrt(sum(v.square().sum() for v in g));factor=.01*torch.clamp(1/(norm+1e-6),max=1)
   if variant=="initial_normmatched":factor=matching[t]/norm.clamp_min(1e-20)
   proposed=float((factor*norm).detach())
   with torch.no_grad():
    for p,v,gg in zip(params,initial,g):p.copy_(v-factor*gg)
    logits=forward.run(x);pr=logits.softmax(1).cpu().numpy();after=-logits.log_softmax(1)[torch.arange(5,device="cuda"),y];actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))))
   assert proposed<=.010001 and actual<=.01001;baseerr=float(abs(bp[1:]-np.array(reference[t]["future_base_probabilities"])).max());assert baseerr<2e-5
   replay=None
   if variant=="initial":replay=float(abs(pr[1:]-np.array(reference[t]["future_write_probabilities"])).max());assert replay<2e-5
   rows.append(dict(variant=variant,subject=sid,t=t,aux_loss=float(aux.detach()),gradient_norm=float(norm.detach()),proposed_update_norm=proposed,actual_update_norm=actual,future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_future_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),current_ce_gain=float(before[0]-after[0]),future_labels=y[1:].cpu().tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=pr[1:].tolist(),base_correct=int((bp[1:].argmax(1)==y[1:].cpu().numpy()).sum()),write_correct=int((pr[1:].argmax(1)==y[1:].cpu().numpy()).sum()),future_argmax_changes=int((bp[1:].argmax(1)!=pr[1:].argmax(1)).sum()),classifier_base_probability_error=baseerr,initial_replay_probability_error=replay))
  restore();assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items());dest.write_text(json.dumps(dict(variant=variant,subject=sid,held_subject_excluded_from_meta_training=True,model_restored_exact=True,rows=rows),indent=2));allrows+=len(rows);print("META HELD PULSES",variant,sid,len(rows),flush=True)
 (out/"complete.json").write_text(json.dumps(dict(variant=variant,rows=allrows),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--validate",action="store_true");parser.add_argument("--variant");a=parser.parse_args()
 if a.validate:validate()
 else:run(a.variant)

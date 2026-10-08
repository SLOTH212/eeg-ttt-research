from phase69_pulses import *
import pandas as pd
OUT75=ROOT/"phase75"
def run():
 OUT75.mkdir(exist_ok=True);df=pd.read_csv(ROOT/"phase74/layers.csv");ce=df[(df.objective=="waveform")&(df.target=="future_ce")].set_index("layer");bal=df[(df.objective=="waveform")&(df.target=="future_balanced_ce")].set_index("layer");eligible=[int(l) for l in ce.index if ce.loc[l,"positive_people"]>=3 and bal.loc[l,"positive_people"]>=3 and ce.loc[l,"mean_contribution"]>0 and bal.loc[l,"mean_contribution"]>0];selected=max(eligible,key=lambda l:ce.loc[l,"mean_contribution"]);assert selected==0
 used=set()
 for sid in [7,17,27,37]:
  for t in np.load(ROOT/"phase74"/f"s{sid:02d}_gradients.npz")["times"]:
   for j in range(1,5):used.add((sid,int(t)+j))
 protocol=dict(scope_selection_people=[7,17,27,37],validation_people=[5,15,25,35],eligible_layers=eligible,selected_layer=selected,selection_rule="ordinary and balanced future contributions positive >=3/4 each; choose largest ordinary mean among eligible",selection_future_class_labels=len(used),lr=.01,variants=["selected_layer","all","all_normmatched_selected"],horizon=4,primary="selected_layer",advance_rule="future CE>0, >=3/4 positive, balanced CE>=0, corrected>=harmed; no automatic target rollout",source_validation_previously_seen=True,scope="posthoc architecture-specific source replication; original full-layer primary stays failed")
 (OUT75/"protocol.json").write_text(json.dumps(protocol,indent=2));m=make("pretrained");r=np.load(ROOT/"phase72/readout.npz")
 with torch.no_grad():m.encoder.proj_out[0].weight.copy_(torch.tensor(r["coef"],device="cuda"));m.encoder.proj_out[0].bias.copy_(torch.tensor(r["bias"],device="cuda"))
 named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];params=[p for n,p in named];active=[int(n.split(".")[3])==selected for n,p in named];initial=[p.detach().clone() for p in params];state=snapshot(m);cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],device="cuda",dtype=torch.float64)
 def restore():
  with torch.no_grad():
   for p,v in zip(params,initial):p.copy_(v)
 for sid in protocol["validation_people"]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");yy=np.load(root/f"s{sid:02d}_meta.npz")["y"];valid=np.array([t for t in range(len(yy)-4) if (yy[t:t+5]>=0).all()]);times=valid[np.linspace(0,len(valid)-1,64,dtype=int)];rows=[]
  for t0 in times:
   t=int(t0);restore();x,y=sparse(sid,t)
   with torch.no_grad():blp=m(x).double().log_softmax(1);bp=blp.exp().cpu().numpy();before=-blp[torch.arange(5,device="cuda"),y]
   gen=torch.Generator(device="cpu").manual_seed(75000+sid*100000+t);mask=(torch.rand((1,2,30),generator=gen)<.5).cuda();pred=m.encoder(x[:1],mask);loss=((pred-x[:1]).square()*mask[...,None]).sum()/(mask.sum()*200);g=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(v.square().sum() for v in g));snorm=torch.sqrt(sum(v.square().sum() for v,a in zip(g,active) if a));clip=torch.clamp(1/(norm+1e-6),max=1.);sclip=torch.clamp(1/(snorm+1e-6),max=1.)
   for variant in protocol["variants"]:
    restore();mult=sclip if variant=="selected_layer" else clip
    if variant=="all_normmatched_selected":mult=sclip*snorm/norm.clamp_min(1e-20)
    with torch.no_grad():
     for p,v,gg,a in zip(params,initial,g,active):
      if variant!="selected_layer" or a:p.copy_(v-.01*mult*gg)
     actual=float(torch.sqrt(sum((p-v).square().sum() for p,v in zip(params,initial))));assert actual<=.01+1e-5
     if variant=="selected_layer":assert all(torch.equal(p,v) for p,v,a in zip(params,initial,active) if not a)
     plp=m(x).double().log_softmax(1);wp=plp.exp().cpu().numpy();after=-plp[torch.arange(5,device="cuda"),y]
    b=bp[1:].argmax(1);w=wp[1:].argmax(1);truth=y[1:].cpu().numpy();rows.append(dict(subject=sid,t=t,variant=variant,lr=.01,future_ce_gain=float((before[1:]-after[1:]).mean()),balanced_ce_gain=float(((before[1:]-after[1:])*cw[y[1:]]).mean()),future_labels=truth.tolist(),future_base_probabilities=bp[1:].tolist(),future_write_probabilities=wp[1:].tolist(),actual_update_norm=actual,corrected=int(((b!=truth)&(w==truth)).sum()),harmed=int(((b==truth)&(w!=truth)).sum()),argmax_changes=int((b!=w).sum())))
  restore();assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items());(OUT75/f"s{sid:02d}.json").write_text(json.dumps(dict(subject=sid,times=times.tolist(),state_restored_exact=True,unselected_layers_exact=True,rows=rows),indent=2));print("HELD SCOPE WRITES",sid,len(rows),flush=True)
 (OUT75/"complete.json").write_text(json.dumps(dict(complete=True,rows=768,people=4,selected_parameters=800)))
if __name__=="__main__":run()

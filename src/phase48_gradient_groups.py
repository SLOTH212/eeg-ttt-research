from phase47_pulses import model_for
from phase46_spectral_head import *
OUT48=ROOT/"phase48"
def run(mode):
 out=OUT48/mode;out.mkdir(exist_ok=True)
 if (out/"complete.json").exists():return
 model=model_for(mode);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named];names=[n for n,p in named];group_names=[f"block{i}" for i in range(12)]+["final_norm"];group=[];layout=[];offset=0
 for n,p in named:
  g=int(n.split(".")[2]) if n.startswith("encoder.blocks.") else 12 if n.startswith("encoder.norm.") else None;assert g is not None;group.append(g);layout.append(dict(name=n,shape=list(p.shape),offset=offset,length=p.numel(),group=g));offset+=p.numel()
 assert offset==10960 and set(group)==set(range(13));meta=dict(group_names=group_names,parameters=layout,total=offset);path=OUT48/"parameter_layout.json"
 if path.exists():assert json.loads(path.read_text())==meta
 else:path.write_text(json.dumps(meta,indent=2))
 state={k:v.clone() for k,v in model.state_dict().items()};cw=torch.tensor(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"],dtype=torch.float32,device="cuda");allrows=[];errors=[];start=time.perf_counter()
 for sid in [3,4,7,12,13,14,17,27,37]:
  done=out/f"s{sid:02d}.json"
  if done.exists():allrows.extend(json.loads(done.read_text())["rows"]);continue
  ref=json.loads((ROOT/"phase47"/mode/f"s{sid:02d}.json").read_text());reference={(r["t"],r["objective"]):r for r in ref["rows"] if r["lr"]==.001};times=sorted(set(r["t"] for r in ref["rows"]));root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");data=np.load(root/f"s{sid:02d}_meta.npz");y=data["y"];features=data["features"][:,24:48];rows=[];gradients={k:[] for k in (["spectral","entropy"] if mode=="masked" else ["spectral"])}
  for t in times:
   xx=preprocess(np.array(raw[t:t+5])).reshape(5,2,2,15,200).cuda();yy=torch.tensor(y[t:t+5],device="cuda");vec=nn.functional.cross_entropy(model(xx[1:]),yy[1:],reduction="none");future=torch.autograd.grad(vec.mean(),params,retain_graph=True);balanced=torch.autograd.grad((vec*cw[yy[1:]]).mean(),params);current=torch.autograd.grad(nn.functional.cross_entropy(model(xx[:1]),yy[:1]),params)
   for objective in gradients:
    if objective=="spectral":loss=model.reconstruction_loss(xx[:1],torch.tensor(features[t:t+1],device="cuda"))
    else:
     lp=model(xx[:1]).log_softmax(1);loss=-(lp.exp()*lp).sum()
    aux=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(v.square().sum() for v in aux));clip=min(1.,1/max(float(norm),1e-12));gradients[objective].append(torch.cat([g.detach().flatten() for g in aux]).cpu().numpy());total=0.
    for group_id,name in enumerate(group_names):
     ii=[i for i,g in enumerate(group) if g==group_id];gn=float(sum(aux[i].square().sum() for i in ii));fn=float(sum(future[i].square().sum() for i in ii));dot=float(sum((aux[i]*future[i]).sum() for i in ii));bdot=float(sum((aux[i]*balanced[i]).sum() for i in ii));cdot=float(sum((aux[i]*current[i]).sum() for i in ii));total+=clip*dot;rows.append(dict(mode=mode,subject=sid,t=t,objective=objective,group=group_id,group_name=name,auxiliary_gradient_norm=float(norm),clip=clip,group_gradient_squared_norm=gn,future_task_squared_norm=fn,future_dot=dot,balanced_future_dot=bdot,current_dot=cdot,balanced_current_dot=float(cw[yy[0]])*cdot,clipped_future_dot=clip*dot,clipped_current_dot=clip*cdot,current_class=int(y[t]),future_labels=y[t+1:t+5].tolist()))
    expected=reference[t,objective]["first_order_gain"]/.001;err=abs(total-expected);assert err<max(1e-5,abs(expected)*1e-4);errors.append(err)
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());np.savez_compressed(out/f"s{sid:02d}_gradients.npz",times=np.array(times),**{k:np.array(v,dtype=np.float32) for k,v in gradients.items()});done.write_text(json.dumps(dict(mode=mode,subject=sid,parameters_unchanged=True,gradient_scope="Current auxiliary input only; future task gradients only label diagnostic dots",rows=rows),indent=2));allrows.extend(rows);print("GROUP GRADIENTS",mode,sid,len(times),len(rows),flush=True)
 pd.DataFrame(allrows).to_csv(out/"groups.csv",index=False);(out/"complete.json").write_text(json.dumps(dict(mode=mode,group_rows=len(allrows),parameters_unchanged=True,max_total_dot_error=max(errors,default=0.),seconds=time.perf_counter()-start),indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True);a=parser.parse_args();run(a.mode)

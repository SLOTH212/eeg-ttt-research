from phase52_evaluate import load
from phase29_forward_graph import LogitForward
from phase46_spectral_head import *
from phase2_temporal import ema
o=ROOT/"phase55";model=load("initial");model.requires_grad_(False);state={n:v.clone() for n,v in model.state_dict().items()};x0=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r");forward=LogitForward(model,torch.tensor(np.array(x0[:1]),device="cuda"));rows=[]
for sid in [5,15,25,35]:
 x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");ref=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz");pred=[]
 with torch.no_grad():
  for t in range(len(x)):pred.append(forward.run(torch.tensor(np.array(x[t:t+1]),device="cuda")).softmax(1).cpu().numpy()[0].copy())
 pred=np.array(pred);error=float(abs(pred-ref["p"]).max());dis=int((pred.argmax(1)!=ref["p"].argmax(1)).sum());assert error<2e-5 and dis==0
 emadis={str(a):int((ema(pred,a).argmax(1)!=ema(ref["p"],a).argmax(1)).sum()) for a in [.8,.5]};assert not any(emadis.values());assert all(torch.equal(v,model.state_dict()[n]) for n,v in state.items());rows.append(dict(subject=sid,windows=len(x),probability_error=error,argmax_disagreements=dis,ema_argmax_disagreements=emadis,all_model_state_unchanged_exact=True));print("FULL FROZEN PATH",rows[-1],flush=True)
(o/"full_zero_forward_audit.json").write_text(json.dumps(dict(status="PASS",reason="Zero-LR updates already audited exactunchanged; full batch1 classifier path matches cached batched frozen predictions, ruling out precision artifacts for tiny-stepF1",cases=rows),indent=2))

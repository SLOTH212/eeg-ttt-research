from phase21_full_encoder import *
import pandas as pd
OUT28=ROOT/"phase28"
if __name__=="__main__":
 splits=json.loads((ROOT/"phase3/protocol.json").read_text())["folds"];rows=[]
 for source in ["young","mixed"]:
  counts=np.ones(8192,dtype=np.float64);position=np.ones((30,8192),dtype=np.float64)
  for sid in splits[source]["base_train"]:
   root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];tokens=np.load(OUT28/"tokens"/f"s{sid:02d}.npz")["tokens"][y>=0].reshape(-1,30);counts+=np.bincount(tokens.ravel(),minlength=8192)
   for j in range(30):position[j]+=np.bincount(tokens[:,j],minlength=8192)
  prior=counts/counts.sum();pp=position/position.sum(1,keepdims=True);np.savez_compressed(OUT28/f"{source}_code_priors.npz",global_prior=prior,position_prior=pp)
  for sid in splits[source]["base_val"]:
   tokens=np.load(OUT28/"tokens"/f"s{sid:02d}.npz")["tokens"];idx=np.linspace(0,len(tokens)-1,min(256,len(tokens)),dtype=int);targets=tokens[idx].reshape(-1,30)[:,::2];pindex=np.arange(0,30,2)[None,:]
   rows.append(dict(source=source,subject=sid,uniform_ce=float(np.log(8192)),global_prior_ce=float(-np.log(prior[targets]).mean()),position_prior_ce=float(-np.log(pp[pindex,targets]).mean()),global_majority_accuracy=float(np.mean(targets==prior.argmax())),position_majority_accuracy=float(np.mean(targets==pp.argmax(1)[None,::2]))))
 pd.DataFrame(rows).to_csv(OUT28/"code_prior_baselines.csv",index=False);print(pd.DataFrame(rows).to_string(index=False))

from phase72_fit import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
def run():
 assert (OUT72/"fitting_complete.json").exists();protocol=json.loads((OUT72/"protocol.json").read_text());choice=protocol["checkpoint"];m=Carrier("pretrained").cuda().eval();m.load_state_dict(torch.load(Path(choice["folder"])/"model.pt",map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);initial=snapshot(m);readout=np.load(OUT72/"readout.npz")
 with torch.no_grad():m.encoder.proj_out[0].weight.copy_(torch.tensor(readout["coef"],device="cuda"));m.encoder.proj_out[0].bias.copy_(torch.tensor(readout["bias"],device="cuda"))
 assert all(torch.equal(v.cpu(),initial[n]) for n,v in m.state_dict().items() if n not in {"encoder.proj_out.0.weight","encoder.proj_out.0.bias"});state=snapshot(m)
 # Independent weighted synthetic regression verifies normalization / intercept algebra.
 rng=np.random.default_rng(72);x=rng.normal(size=(301,20));y=rng.normal(size=(301,11));w=rng.uniform(.1,2,size=301);w/=w.mean();sc=StandardScaler().fit(x,sample_weight=w);fit=Ridge(alpha=100).fit(sc.transform(x),y,sample_weight=w)
 mx=np.average(x,axis=0,weights=w);my=np.average(y,axis=0,weights=w);cov=((x-mx)*w[:,None]).T@(x-mx)/len(x);cross=((x-mx)*w[:,None]).T@(y-my)/len(x);scale=np.sqrt(np.diag(cov));coef=np.linalg.solve(cov/scale[:,None]/scale[None,:]+100/len(x)*np.eye(20),cross/scale[:,None]).T
 error=float(abs(coef-fit.coef_).max());assert error<1e-12
 (OUT72/"regression_algebra_audit.json").write_text(json.dumps(dict(status="PASS",independent_sklearn_weighted_ridge_coefficient_error=error,samples=301,features=20,targets=11,alpha=100),indent=2))
 rows=[];checks=[];(OUT72/"cases").mkdir(exist_ok=True)
 def predict(x,mask):
  chunks=[]
  with torch.no_grad():
   for begin in range(0,len(x),8):chunks.append(m.encoder(torch.tensor(x[begin:begin+8],device="cuda"),torch.tensor(mask[begin:begin+8],device="cuda")).cpu().numpy())
  return np.concatenate(chunks)
 for sid in [5,15,25,35]:
  x=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_input.npz")["x"];old=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_bernoulli.npz");post=np.load(ROOT/"phase68/cases"/f"pretrained_s{sid:02d}.npz");mask=old["mask"];p=predict(x,mask);sh=predict(np.roll(x,48,axis=0),mask);template=(x*(~mask)[...,None]).sum(2)/(~mask).sum(2)[...,None];values=[]
  for i in range(len(x)):
   target=x[i][mask[i]].astype(np.float64);pred=p[i][mask[i]].astype(np.float64);sp=sh[i][mask[i]].astype(np.float64);vp=np.broadcast_to(template[i,:,None,:],x[i].shape)[mask[i]].astype(np.float64);values.append([np.square(pred-target).mean(),np.square(target).mean(),np.square(sp-target).mean(),np.square(vp-target).mean()])
  values=np.array(values);mean=values.mean(0);assert np.isfinite(values).all()
  with torch.no_grad():
   xt=torch.tensor(x[:2],device="cuda");mt=torch.tensor(mask[:2],device="cuda");base=m.encoder(xt,mt).clone();changed=xt.clone();changed[mt]+=3.17;leak=float(abs(m.encoder(changed,mt)-base).max());assert leak==0
   z=latent(m,xt,mt).cpu().numpy().astype(np.float64);reference=z@readout["coef"].astype(np.float64).T+readout["bias"];linearerr=float(abs(reference-base.cpu().numpy()).max());assert linearerr<2e-5
  np.savez_compressed(OUT72/"cases"/f"s{sid:02d}.npz",mask=mask,masked_target=x[mask],masked_prediction=p[mask],masked_shuffle_prediction=sh[mask],window_metrics=values)
  rows.append(dict(subject=sid,mse=mean[0],zero_mse=mean[1],shuffled_mse=mean[2],visible_template_mse=mean[3],original_frozen_pretrained_mse=float(old["window_metrics"][:,0].mean()),unrefitted_finetuned_mse=float(post["window_metrics"][:,0].mean()),ratio_to_zero=mean[0]/mean[1],validity_pass=bool(mean[0]<mean[1] and mean[0]<mean[2])))
  checks.append(dict(subject=sid,raw_masked_perturbation_error=leak,torch_linear_vs_numpy_error=linearerr));print("REFITTED WAVEFORM VALIDITY",rows[-1],flush=True)
 assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 df=pd.DataFrame(rows);df.to_csv(OUT72/"validation.csv",index=False);decision=dict(validity_pass=bool(df.validity_pass.all()),passing_people=int(df.validity_pass.sum()),people=4,extra_class_labels=0,extra_parameters_fitted=40200,classification_path_exact_unchanged=True,no_ttt_claim=True);(OUT72/"decision.json").write_text(json.dumps(decision,indent=2));(OUT72/"execution_audit.json").write_text(json.dumps(dict(status="PASS",cases=checks,postfit_model_unchanged=True,only_projection_was_refitted=True),indent=2));print(decision)
if __name__=="__main__":run()

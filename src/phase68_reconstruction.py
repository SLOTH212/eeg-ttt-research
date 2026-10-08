from phase66_train import *
import pandas as pd
OUT68=ROOT/"phase68"
def run():
 selection=json.loads((OUT66/"selection.json").read_text())["selected"];rows=[];audits=[];(OUT68/"cases").mkdir(exist_ok=True)
 for kind,chosen in selection.items():
  folder=Path(chosen["folder"]);assert hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()==chosen["checkpoint_sha256"]
  model=Carrier(kind).cuda().eval();model.load_state_dict(torch.load(folder/"model.pt",map_location="cpu",weights_only=False)["state"]);model.requires_grad_(False);state=snapshot(model)
  original=torch.load(ROOT/"phase63"/(kind+"_encoder_state.pt"),map_location="cpu",weights_only=True);assert all(torch.equal(v.cpu(),original["proj_out."+n]) for n,v in model.encoder.proj_out.state_dict().items())
  def predict(x,mask):
   result=[]
   with torch.no_grad():
    for start in range(0,len(x),8):result.append(model.encoder(torch.tensor(x[start:start+8],device="cuda"),torch.tensor(mask[start:start+8],device="cuda")).cpu().numpy())
   return np.concatenate(result)
  for sid in [5,15,25,35]:
   inp=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_input.npz");old=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_bernoulli.npz");x=inp["x"];times=inp["times"];mask=old["mask"];pred=predict(x,mask);shuffled=predict(np.roll(x,48,axis=0),mask);values=[]
   for i in range(len(x)):
    y=x[i][mask[i]].astype(np.float64);p=pred[i][mask[i]].astype(np.float64);s=shuffled[i][mask[i]].astype(np.float64);values.append([np.square(p-y).mean(),np.square(y).mean(),np.square(s-y).mean()])
   values=np.array(values);assert np.isfinite(values).all();mse,zero,sh=values.mean(0)
   with torch.no_grad():
    xt=torch.tensor(x[:2],device="cuda");mt=torch.tensor(mask[:2],device="cuda");base=model.encoder(xt,mt).clone();altered=xt.clone();altered[mt]+=3.17;leak=float(abs(model.encoder(altered,mt)-base).max());assert leak==0
    probs=model(torch.tensor(x,device="cuda")).softmax(1).cpu().numpy()
   reference=np.load(folder/f"s{sid:02d}.npz")["p"][times];err=float(abs(probs-reference).max());assert err<2e-5 and np.array_equal(probs.argmax(1),reference.argmax(1))
   assert np.array_equal(x[mask],old["masked_target"])
   np.savez_compressed(OUT68/"cases"/f"{kind}_s{sid:02d}.npz",times=times,mask=mask,masked_target=x[mask],masked_prediction=pred[mask],masked_shuffle_prediction=shuffled[mask],window_metrics=values)
   rows.append(dict(kind=kind,subject=sid,classification_best_epoch=chosen["best_epoch"],masked_mse=mse,zero_mse=zero,shuffled_mse=sh,ratio_to_zero=mse/zero,ratio_to_shuffle=mse/sh,original_pretrained_mse=float(old["window_metrics"][:,0].mean()),validity_pass=bool(mse<zero and mse<sh)))
   audits.append(dict(kind=kind,subject=sid,classifier_probability_error=err,classifier_argmax_exact=True,masked_perturbation_error=leak,masked_targets_exact=True));print("POST FINETUNE RECON",rows[-1],flush=True)
  assert all(torch.equal(v.cpu(),state[n]) for n,v in model.state_dict().items());del model,state
 df=pd.DataFrame(rows);df.to_csv(OUT68/"reconstruction.csv",index=False);decision=dict(primary_pretrained_valid=bool(df[df.kind=="pretrained"].validity_pass.all()),passing_people=int(df[df.kind=="pretrained"].validity_pass.sum()),people=4,classifier_scope="source-finetuned carrier",no_auxiliary_refitting=True,no_online_claim=True)
 (OUT68/"decision.json").write_text(json.dumps(decision,indent=2));(OUT68/"execution_audit.json").write_text(json.dumps(dict(status="PASS",cases=audits,full_state_unchanged=True,projection_unchanged=True,selected_checkpoint_hashes_verified=True),indent=2));print(decision)
if __name__=="__main__":run()

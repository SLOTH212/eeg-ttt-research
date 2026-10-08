from phase62_reconstruction import ROOT,prep,CBraMod,torch,np,json,time,os
OUT=ROOT/"phase63"
def features(model,x):
 return model.encoder(model.patch_embedding(x)).mean(2).flatten(1)
def run():
 pr=json.loads((OUT/"protocol.json").read_text());subjects=pr["training_subjects"]+pr["validation_subjects"]
 for kind in ["pretrained","random"]:
  torch.manual_seed(17);model=CBraMod().cuda().eval()
  if kind=="pretrained":model.load_state_dict(torch.load(ROOT/"phase62/pretrained_weights.pth",map_location="cpu",weights_only=True),strict=True)
  model.requires_grad_(False);state={n:v.cpu().clone() for n,v in model.state_dict().items()};torch.save(state,OUT/(kind+"_encoder_state.pt"))
  for sid in subjects:
   done=OUT/"features"/f"s{sid:02d}_{kind}.json"
   if done.exists():continue
   folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");parts=[];start=time.perf_counter();audit={}
   with torch.no_grad():
    for begin in range(0,len(raw),32):
     array=prep(np.array(raw[begin:begin+32]),[.1,45]);x=torch.tensor(array,device="cuda");z=features(model,x);parts.append(z.cpu().numpy())
     if begin==0:
      single=features(model,x[:1]);err=float(abs(single-z[:1]).max());relative=float((single-z[:1]).norm()/z[:1].norm().clamp_min(1e-12));assert err<2e-4 and relative<2e-5
      altered=x.clone();altered[1:]=altered[1:]*1.7+.3;cross=float(abs(features(model,altered)[:1]-z[:1]).max());assert cross<2e-5
      assert np.array_equal(prep(np.array(raw[:1]),[.1,45]),array[:1])
      manual=model.proj_out(model.encoder(model.patch_embedding(x[:1])));whole=model(x[:1]);headerr=float(abs(manual-whole).max());assert headerr==0
      audit=dict(batch_one_max_error=err,batch_one_relative_error=relative,other_sample_error=cross,perwindow_preprocessing_exact=True,full_model_decomposition_error=headerr)
   z=np.concatenate(parts);assert z.shape==(len(raw),400) and np.isfinite(z).all()
   assert all(torch.equal(v.cpu(),state[n]) for n,v in model.state_dict().items())
   np.savez_compressed(OUT/"features"/f"s{sid:02d}_{kind}.npz",features=z)
   rec=dict(subject=sid,kind=kind,shape=list(z.shape),audit=audit,encoder_state_unchanged=True,seconds=time.perf_counter()-start)
   done.write_text(json.dumps(rec,indent=2));print("CBRAMOD FEATURES",kind,sid,rec["seconds"],flush=True)
  del model,state
 (OUT/"features_complete.json").write_text(json.dumps(dict(complete=True,people=len(subjects),models=2)))
if __name__=="__main__":run()

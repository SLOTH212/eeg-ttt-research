from phase66_train import *
OUT72=ROOT/"phase72"
def latent(m,x,mask):return m.encoder.encoder(m.encoder.patch_embedding(x,mask))
def run():
 protocol=json.loads((OUT72/"protocol.json").read_text());choice=protocol["checkpoint"];path=Path(choice["folder"])/"model.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==choice["checkpoint_sha256"]
 m=Carrier("pretrained").cuda().eval();m.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state"]);m.requires_grad_(False);state=snapshot(m);stats=[];total=0;counts={}
 for sid in protocol["training_subjects"]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");rng=torch.Generator().manual_seed(72000+sid);mask=(torch.rand((len(raw),2,30),generator=rng)<.5).numpy();n=int(mask.sum());total+=n;counts[sid]=dict(windows=len(raw),masked_patches=n);sums=np.zeros(400,np.float64);xx=np.zeros((200,200),np.float64);xy=np.zeros((200,200),np.float64);start=time.perf_counter();observed=0
  with torch.no_grad():
   for begin in range(0,len(raw),64):
    array=prep(np.array(raw[begin:begin+64]),[.1,45]);mk=mask[begin:begin+64];xt=torch.tensor(array,device="cuda");mt=torch.tensor(mk,device="cuda");z=latent(m,xt,mt)[mt].cpu().numpy().astype(np.float64);y=array[mk].astype(np.float64)
    sums[:200]+=z.sum(0);sums[200:]+=y.sum(0);xx+=z.T@z;xy+=z.T@y;observed+=len(z)
  assert observed==n;stats.append((sums[:200]/n,sums[200:]/n,xx/n,xy/n));print("WAVEFORM READOUT STATS",sid,n,time.perf_counter()-start,flush=True)
 meanx=np.mean([s[0] for s in stats],0);meany=np.mean([s[1] for s in stats],0);covxx=np.mean([s[2] for s in stats],0)-np.outer(meanx,meanx);covxy=np.mean([s[3] for s in stats],0)-np.outer(meanx,meany);variance=np.diag(covxx);assert variance.min()>0;scale=np.sqrt(variance);a=covxx/scale[:,None]/scale[None,:]+100/total*np.eye(200);b=covxy/scale[:,None];coef_standard=np.linalg.solve(a,b).T;coef=coef_standard/scale[None,:];bias=meany-coef@meanx;residual=float(np.linalg.norm(a@coef_standard.T-b)/max(np.linalg.norm(b),1e-30));assert residual<1e-10
 np.savez(OUT72/"readout.npz",coef=coef.astype(np.float32),bias=bias.astype(np.float32),feature_mean=meanx,feature_scale=scale,target_mean=meany,covariance=covxx,cross_covariance=covxy,standardized_coef=coef_standard,total_masked_patches=total,alpha=100.)
 assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items())
 with torch.no_grad():m.encoder.proj_out[0].weight.copy_(torch.tensor(coef,dtype=torch.float32,device="cuda"));m.encoder.proj_out[0].bias.copy_(torch.tensor(bias,dtype=torch.float32,device="cuda"))
 allowed={"encoder.proj_out.0.weight","encoder.proj_out.0.bias"};assert all(torch.equal(v.cpu(),state[n]) for n,v in m.state_dict().items() if n not in allowed)
 original=Carrier("pretrained").cuda().eval();original.load_state_dict(state);original.requires_grad_(False);predchecks=[]
 for sid in protocol["validation_subjects"]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");saved=np.load(Path(choice["folder"])/f"s{sid:02d}.npz")["p"];worst=0.
  with torch.no_grad():
   for begin in range(0,len(raw),128):
    x=torch.tensor(prep(np.array(raw[begin:begin+128]),[.1,45]),device="cuda");before=original(x);after=m(x);assert torch.equal(before,after);p=after.softmax(1).cpu().numpy();worst=max(worst,float(abs(p-saved[begin:begin+len(x)]).max()));assert np.array_equal(p.argmax(1),saved[begin:begin+len(x)].argmax(1))
  assert worst<2e-5;predchecks.append(dict(subject=sid,windows=len(raw),before_after_logits_exact=True,saved_prediction_error=worst))
 (OUT72/"fitting_complete.json").write_text(json.dumps(dict(complete=True,source_person_counts=counts,source_windows=sum(v["windows"] for v in counts.values()),masked_patches=total,fit_parameters=40200,normal_equation_relative_residual=residual,classification_labels_not_read=True,nonprojection_state_exact=True,classification_checks=predchecks,checkpoint_sha256=choice["checkpoint_sha256"]),indent=2));print("WAVEFORM READOUT FIT COMPLETE",total,residual,flush=True)
if __name__=="__main__":run()

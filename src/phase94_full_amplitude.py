from phase66_train import *
O=ROOT/"phase94";O.mkdir(exist_ok=True)
def prepare_views(sid,gains):
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");views=[]
 for gain in gains:
  parts=[]
  for i in range(0,len(raw),128):parts.append(prep(np.clip(np.array(raw[i:i+128])*gain,-20,20).astype(np.float32),[.1,45]))
  views.append(np.concatenate(parts))
 return torch.from_numpy(np.stack(views,axis=1)),np.load(folder/f"s{sid:02d}_meta.npz")["y"]
def run():
 assert json.loads((ROOT/"phase93/audit.json").read_text())["status"]=="PASS";old=json.loads((ROOT/"phase93/protocol.json").read_text());train=old["train"];val=old["validation"];path=Path(old["checkpoint"])
 pr=dict(purpose="Full-encoder static amplitude augmentation after fixed-encoder head-only source guard failure; separate exploratory branch, prior failures preserved.",train=train,validation=val,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gains=old["gains"],input=old["input"],methods=old["methods"],epochs=5,batch=64,encoder_lr=2e-5,head_lr=1e-4,weight_decay=.01,gradient_clip=1.,trainable="All classification encoder and classifier parameters; unused waveform projection frozen.",selection=old["selection"].replace("epoch0..10","epoch0..5"),epoch_range="epoch0..5 (five fixed continuation epochs)",primary=old["primary"],advance=old["advance"],weights=old["weights"],matching="Both branches same3views/window label presentations, dedicatedorder17 and dropoutseed17; difference only whether gains0.1/1/10 or repeatednative input.",label_scope="Existing10source training people only; no newunique labels. Source4validation reused forselection; all external designs exploratory after prior target exposure.",cpu_ids=list(range(8,24)),gpu=0)
 p=O/"protocol.json"
 if p.exists():assert json.loads(p.read_text())==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 parts=[];ys=[];ss=[]
 for sid in train:
  x,y=prepare_views(sid,pr["gains"]);good=y>=0;parts.append(x[good]);ys.append(y[good]);ss.extend([sid]*int(good.sum()));print("PREP TRAIN",sid,int(good.sum()),flush=True)
 x=torch.cat(parts);y=torch.tensor(np.concatenate(ys));del parts;ss=np.array(ss);counts=torch.bincount(y,minlength=5).numpy();cw=np.sqrt(len(y)/np.maximum(counts,1));nper={sid:int((ss==sid).sum()) for sid in train};w=np.array([1/nper[int(s)] for s in ss])*cw[y.numpy()];w=torch.tensor(w/w.mean(),dtype=torch.float32)
 validation={sid:prepare_views(sid,pr["gains"]) for sid in val};print("PREP COMPLETE",tuple(x.shape),flush=True)
 base=torch.load(path,map_location="cpu",weights_only=False)["state"]
 selections={};histories={}
 for method in pr["methods"]:
  out=O/method;out.mkdir(exist_ok=True)
  if (out/"complete.json").exists():
   d=json.loads((out/"complete.json").read_text());selections[method]=d["selection"];histories[method]=d["permutation_hashes"];continue
  m=Carrier("pretrained").cuda();m.load_state_dict(base);m.requires_grad_(True);m.encoder.proj_out.requires_grad_(False);projection={k:v.detach().cpu().clone() for k,v in m.encoder.proj_out.state_dict().items()}
  def evaluate():
   m.eval();scores=[]
   for sid,(vx,vy) in validation.items():
    for j,gain in enumerate(pr["gains"]):scores.append(dict(subject=sid,gain=gain,macro_f1=100*metrics(vy,probabilities(m,vx[:,j]))["macro_f1"]))
   return scores
  baseline=evaluate();native={v["subject"]:v["macro_f1"] for v in baseline if v["gain"]==1};bn=float(np.mean(list(native.values())));br=float(np.mean([v["macro_f1"] for v in baseline]));ref=json.loads((ROOT/"phase93/repeat_raw/history.json").read_text())["history"][0];assert abs(br-ref["robust_f1"])<1e-8 and abs(bn-ref["native_f1"])<1e-8
  opt=torch.optim.AdamW([dict(params=[p for p in m.encoder.parameters() if p.requires_grad],lr=pr["encoder_lr"]),dict(params=m.head.parameters(),lr=pr["head_lr"])],weight_decay=pr["weight_decay"]);seed_all(17);gen=torch.Generator(device="cpu").manual_seed(17);best=br;bestepoch=0;beststate=snapshot(m);history=[dict(epoch=0,robust_f1=br,native_f1=bn,eligible=True,person_scores=baseline)];orders=[];start_epoch=0;started=time.perf_counter()
  resume=out/"resume.pt"
  if resume.exists():
   ck=torch.load(resume,map_location="cpu",weights_only=False);m.load_state_dict(ck["state"]);opt.load_state_dict(ck["optimizer"]);best=ck["best"];bestepoch=ck["bestepoch"];beststate=ck["beststate"];history=ck["history"];orders=ck["orders"];start_epoch=ck["epoch"];gen.set_state(ck["order_rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state(ck["cuda_rng"])
  for epoch in range(start_epoch,pr["epochs"]):
   m.train();order=torch.randperm(len(y)*3,generator=gen);orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[];epstart=time.perf_counter()
   for bi,idx in enumerate(order.split(pr["batch"])):
    baseidx=idx//3;view=idx%3 if method=="amplitude_augmented" else torch.ones_like(baseidx);xx=x[baseidx,view].cuda();yy=y[baseidx].cuda();ww=w[baseidx].cuda();opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(m(xx),yy,reduction="none")*ww).mean();loss.backward();norm=nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
    if (bi+1)%300==0:print("BATCH",method,epoch+1,bi+1,round(time.perf_counter()-epstart,1),flush=True)
   scores=evaluate();nat=[v for v in scores if v["gain"]==1];nm=float(np.mean([v["macro_f1"] for v in nat]));robust=float(np.mean([v["macro_f1"] for v in scores]));eligible=nm>=bn-1 and all(v["macro_f1"]>=native[v["subject"]]-2 for v in nat);row=dict(epoch=epoch+1,robust_f1=robust,native_f1=nm,eligible=eligible,person_scores=scores,loss=float(np.mean(losses)),seconds=time.perf_counter()-epstart);history.append(row)
   if eligible and robust>best:best=robust;bestepoch=epoch+1;beststate=snapshot(m)
   assert all(torch.equal(v.cpu(),projection[k]) for k,v in m.encoder.proj_out.state_dict().items())
   save(dict(state=snapshot(m),optimizer=opt.state_dict(),best=best,bestepoch=bestepoch,beststate=beststate,history=history,orders=orders,epoch=epoch+1,order_rng=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
   (out/"history.json").write_text(json.dumps(dict(history=history,permutation_hashes=orders),indent=2));print("FULL AUG",method,row,flush=True)
  save(dict(state=beststate,best_epoch=bestepoch,robust_f1=best,history=history),out/"model.pt");selection=dict(epoch=bestepoch,robust_f1=best,selected_metrics=history[bestepoch],checkpoint_sha256=hashlib.sha256((out/"model.pt").read_bytes()).hexdigest());selections[method]=selection;histories[method]=orders
  (out/"complete.json").write_text(json.dumps(dict(selection=selection,permutation_hashes=orders,training_windows=len(y),label_presentations=len(y)*3*pr["epochs"],unused_projection_unchanged_exact=True,seconds=time.perf_counter()-started),indent=2));print("METHOD COMPLETE",method,selection,flush=True);del m,opt,beststate;torch.cuda.empty_cache()
 assert histories["repeat_raw"]==histories["amplitude_augmented"];advance=selections["amplitude_augmented"]["epoch"]>0 and selections["amplitude_augmented"]["robust_f1"]>selections["repeat_raw"]["robust_f1"];(O/"selection.json").write_text(json.dumps(dict(selection=selections,advance=advance),indent=2));(O/"complete.json").write_text(json.dumps(dict(status="PASS",same_order_and_presentations=True,advance=advance)));print("COMPLETE",advance,flush=True)
if __name__=="__main__":run()

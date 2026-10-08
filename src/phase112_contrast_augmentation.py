from phase66_train import *
O=ROOT/"phase112";O.mkdir(exist_ok=True)
def alter(raw,gain,ratio):
 x=np.asarray(raw,dtype=np.float64);g=np.broadcast_to(np.asarray(gain,dtype=np.float64),(len(x),));r=np.broadcast_to(np.asarray(ratio,dtype=np.float64),(len(x),));z=x*np.stack([np.sqrt(r),1/np.sqrt(r)],axis=1)[:,:,None];e=np.mean(x*x,axis=(1,2));ze=np.mean(z*z,axis=(1,2));scale=np.ones(len(x));valid=ze>1e-30;scale[valid]=np.sqrt(e[valid]/ze[valid]);z*= (scale*g)[:,None,None];err=float(np.max(abs(np.mean(z*z,axis=(1,2))-e*g*g)/np.maximum(e*g*g,1e-30)));assert err<1e-12;return np.clip(z,-20,20).astype(np.float32),dict(max_preclip_energy_error=err,clipped_samples=int((abs(z)>20).sum()),samples=z.size)
def prepare(sid,kind):
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(folder/f"s{sid:02d}_x.npy",mmap_mode="r");n=len(raw);rng=np.random.default_rng(112000+sid);u=rng.uniform(-np.log(10),np.log(10),size=(n,2));conditions=[(g,r) for g in [.1,1.,10.] for r in [.1,1.,10.]] if kind=="validation" else [(1.,1.),(np.exp(u[:,0]),np.exp(u[:,1]) if kind=="contrast" else 1.),(np.exp(-u[:,0]),np.exp(-u[:,1]) if kind=="contrast" else 1.)];views=[];records=[]
 for g,r in conditions:
  gg=np.broadcast_to(np.asarray(g),(n,));rr=np.broadcast_to(np.asarray(r),(n,));parts=[];clipped=0;error=0.
  for i in range(0,n,128):
   xx,stats=alter(raw[i:i+128],gg[i:i+128],rr[i:i+128]);parts.append(prep(xx,[.1,45]));clipped+=stats["clipped_samples"];error=max(error,stats["max_preclip_energy_error"])
  views.append(np.concatenate(parts));records.append(dict(max_energy_error=error,clipped_samples=clipped,total_samples=n*6000))
 return torch.from_numpy(np.stack(views,axis=1)),np.load(folder/f"s{sid:02d}_meta.npz")["y"],records
def run():
 old=json.loads((ROOT/"phase94/protocol.json").read_text());path=Path(old["checkpoint"]);train=old["train"];val=old["validation"];methods=["repeat_raw","common_only","common_plus_contrast"]
 pr=dict(purpose="Source-only seed17 pilot: isolate channelrelativegain robustness from commonamplitude, matching preclipjointenergy. Staticfullencodertraining, noTTT.",train=train,validation=val,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),methods=methods,views="3fixedviews/currentwindow: native; g=exp(u),r=exp(v); reciprocalg/r. u,v independentUniform[-ln10,ln10],generator112000+subject beforelabelmask. Commononly uses r=1; contrast uses channelmultiplierssqrt(r),1/sqrt(r) and currentwindow energyrenormalization to samecommonjointRMS. Clippingmaybreakexactpostclipenergymatch and isrecorded.",validation_grid="Fixed9 gain xcontrast pairs {.1,1,10} squared, perwindowjointenergyrenormalized beforeclip.",epochs=5,batch=64,encoder_lr=2e-5,head_lr=1e-4,weight_decay=.01,gradient_clip=1.,weights=old["weights"],selection="Earliestbest meanMacroF1 over4sourcepeople x9fixedconditions amongepoch0..5, eligibleonlyif native mean decline<=1pp andeachperson decline<=2pp versus owninitial. Initialepoch0 fallback.",advance="Contrast selectedepoch>0 androbustsource score>commononly ANDrepeatraw, with unchangednativeguards. Ifnot, no targetrollout or thresholdrelaxation.",matching="Same3labelpresentations/window/epoch, dedicatedorder17/dropout17, sameinitialweights andoptimizer; no newunique labels. Relativecontrast isonlyadditionalperturbation vscommonbranch; comparesa newpairedcommoncontrol, notoldphase94fixed3gainresults.",storage="RAMpreparedviews;selectedandfinalmodels only, nooptimizerresumefiles due limiteddisk.",scope="Previouslyobservedsourcevalidation, exploratorymechanism. Nineviewsarenotindependentpeople. No targetscore usedfor selection.",cpu_ids=list(range(8,24)),gpu=0)
 p=O/"protocol.json"
 if p.exists():assert json.loads(p.read_text())==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest())
 datasets={};prep_audits=[];ys=[];ss=[]
 for kind in ["common","contrast"]:
  parts=[]
  for sid in train:
   xx,yy,audit=prepare(sid,kind);good=yy>=0;parts.append(xx[good]);prep_audits.append(dict(kind=kind,subject=sid,views=audit))
   if kind=="common":ys.append(yy[good]);ss.extend([sid]*int(good.sum()))
  datasets[kind]=torch.cat(parts);del parts;print("PREP",kind,tuple(datasets[kind].shape),flush=True)
 y=torch.tensor(np.concatenate(ys));ss=np.array(ss);counts=torch.bincount(y,minlength=5).numpy();cw=np.sqrt(len(y)/np.maximum(counts,1));nper={s:int((ss==s).sum()) for s in train};w=np.array([1/nper[int(s)] for s in ss])*cw[y.numpy()];w=torch.tensor(w/w.mean(),dtype=torch.float32);validation={}
 for sid in val:
  xx,yy,audit=prepare(sid,"validation");validation[sid]=(xx,yy);prep_audits.append(dict(kind="validation",subject=sid,views=audit))
 (O/"preparation_audit.json").write_text(json.dumps(dict(status="PASS",records=prep_audits),indent=2));base=torch.load(path,map_location="cpu",weights_only=False)["state"];selections={};orders_by_method={};baseref=None
 for method in methods:
  out=O/method;out.mkdir(exist_ok=True)
  if (out/"complete.json").exists():
   completed=json.loads((out/"complete.json").read_text());selections[method]=completed["selection"];orders_by_method[method]=completed["orders"];continue
  m=Carrier("pretrained").cuda();m.load_state_dict(base);m.requires_grad_(True);m.encoder.proj_out.requires_grad_(False);projection={k:v.detach().cpu().clone() for k,v in m.encoder.proj_out.state_dict().items()}
  def evaluate():
   m.eval();scores=[]
   for sid,(vx,vy) in validation.items():
    for j,(gain,ratio) in enumerate([(g,r) for g in [.1,1.,10.] for r in [.1,1.,10.]]):scores.append(dict(subject=sid,gain=gain,ratio=ratio,macro_f1=100*metrics(vy,probabilities(m,vx[:,j]))["macro_f1"]))
   return scores
  baseline=evaluate();native={v["subject"]:v["macro_f1"] for v in baseline if v["gain"]==1 and v["ratio"]==1};bn=float(np.mean(list(native.values())));br=float(np.mean([v["macro_f1"] for v in baseline]));assert abs(bn-66.16565023847805)<1e-5
  if baseref is None:baseref=baseline
  else:assert baseline==baseref
  opt=torch.optim.AdamW([dict(params=[p for p in m.encoder.parameters() if p.requires_grad],lr=pr["encoder_lr"]),dict(params=m.head.parameters(),lr=pr["head_lr"])],weight_decay=pr["weight_decay"]);seed_all(17);gen=torch.Generator().manual_seed(17);best=br;bestepoch=0;beststate=snapshot(m);history=[dict(epoch=0,robust_f1=br,native_f1=bn,eligible=True,person_scores=baseline)];orders=[];data=datasets["contrast" if method=="common_plus_contrast" else "common"]
  for epoch in range(5):
   m.train();order=torch.randperm(len(y)*3,generator=gen);orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());start=time.perf_counter();losses=[]
   for idx in order.split(64):
    bi=idx//3;view=idx%3 if method!="repeat_raw" else torch.zeros_like(bi);xx=data[bi,view].cuda();yy=y[bi].cuda();ww=w[bi].cuda();opt.zero_grad(set_to_none=True);loss=(nn.functional.cross_entropy(m(xx),yy,reduction="none")*ww).mean();loss.backward();norm=nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
   scores=evaluate();native_scores=[r for r in scores if r["gain"]==1 and r["ratio"]==1];nm=float(np.mean([r["macro_f1"] for r in native_scores]));robust=float(np.mean([r["macro_f1"] for r in scores]));eligible=nm>=bn-1 and all(r["macro_f1"]>=native[r["subject"]]-2 for r in native_scores);history.append(dict(epoch=epoch+1,robust_f1=robust,native_f1=nm,eligible=eligible,person_scores=scores,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
   if eligible and robust>best:best=robust;bestepoch=epoch+1;beststate=snapshot(m)
   assert all(torch.equal(v.cpu(),projection[k]) for k,v in m.encoder.proj_out.state_dict().items());(out/"history.json").write_text(json.dumps(dict(history=history,orders=orders),indent=2));print("EPOCH",method,epoch+1,nm,robust,eligible,flush=True)
  save(dict(state=snapshot(m),epoch=5),out/"final5.pt");save(dict(state=beststate,epoch=bestepoch,robust_f1=best),out/"model.pt");selection=dict(epoch=bestepoch,robust_f1=best,native_f1=history[bestepoch]["native_f1"],checkpoint_sha256=hashlib.sha256((out/"model.pt").read_bytes()).hexdigest());selections[method]=selection;orders_by_method[method]=orders;(out/"complete.json").write_text(json.dumps(dict(selection=selection,orders=orders,label_presentations=len(y)*15,training_windows=len(y),projection_frozen_exact=True),indent=2));del m,opt,beststate;torch.cuda.empty_cache();print("BRANCH COMPLETE",method,selection,flush=True)
 assert all(orders_by_method[m]==orders_by_method[methods[0]] for m in methods);advance=selections["common_plus_contrast"]["epoch"]>0 and selections["common_plus_contrast"]["robust_f1"]>max(selections[m]["robust_f1"] for m in ["common_only","repeat_raw"]);(O/"selection.json").write_text(json.dumps(dict(selection=selections,advance=advance),indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",same_initial_weights=True,same_training_orders=True,same_label_presentations=True,native_guards_unchanged=True,advance=advance),indent=2));print("COMPLETE",advance,flush=True)
if __name__=="__main__":run()

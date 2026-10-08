from phase86_meta_waveform import *
from phase66_train import prepared as cbr_prepared
from phase27_train import FullCarrier,prepared as lab_prepared
from phase119_models import ResidualLogitHead,transform
import datetime
O=ROOT/"phase130";O.mkdir(exist_ok=True);C=O/"controller_logits";C.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.inference_mode()
def hp(m,x):
 m.eval();return torch.cat([m(x[a:a+1024]).softmax(1).cpu() for a in range(0,len(x),1024)]).numpy()
def run():
 phases={"cbramod":"phase119","labram":"phase121"};train=[0,1,2,10,11,20,21,30,31,32];original_val=[5,15,25,35];new_val=[7,17,27,37];seeds=[17,29,43];variants=["current","repeat11","history11"]
 specs=[s for s in json.loads((ROOT/"phase79/protocol.json").read_text())["models"] if s["model"] in phases]
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Source checkpoint-selector sensitivity of fixed staticcausalhistory method; swap4sourcevalidation people, do not choosebetterselector fromtargetresults.",training=train,original_selector=original_val,new_selector=new_val,backbone_phases=phases,parent_models=specs,seeds=seeds,variants=variants,training_recipe="Exactlyphase119/121 source10,30epochs,AdamW.001wd.001,batch128,sqrtinverseclassweights,zeroresidualhead,sourcefeaturemean/std floor.1,same seedpermutation/dropout. Onlyselectionpeople change.",selection="Earliestbest new4people rawmeanMacroF1 amongepoch0..30, meanwithin1pp/eachpersonwithin2pp of ownparentbaseline,epoch0fallback. Original4people evaluatedonlyfor historicaltrajectory replay and subsequent sourcetransfer, notnewselection.",reproduction="InitialtrainableparameterSHA and30permutationSHAs exact versus original. Atoldselected andfinal30 comparefullstate maxabs<=5e-5, atoldselected compareoriginalsource savedprobabilities<=2e-5 andargmaxexact. Record everyepoch oldselector MacroF1difference; no newtargetaccess duringtraining.",scope="Allsource/targetpeoplealreadyobserved, notblindvalidation. Parentencoders remaintrainedonoriginal10, so both4person selectors are disjointfromsourcefitting. Evaluateallnewsource-selected outcomes includingfallbacks in laterfixedtarget sensitivity; no selection between original/newselector using targets.",original_protocol_hashes={k:sha(ROOT/v/"protocol.json") for k,v in phases.items()},cpu_ids=list(range(8,24)),gpu=True)
 pp=O/"protocol.json"
 if pp.exists():
  old=json.loads(pp.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:pp.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(pp))
 feature_audits=[]
 for sp in specs:
  backbone=sp["model"];seed=sp["seed"];ip=Path(sp["path"]);assert sha(ip)==sp["sha256"];m=Carrier("pretrained") if backbone=="cbramod" else FullCarrier("mixed","pretrained");m.load_state_dict(torch.load(ip,map_location="cpu",weights_only=False)["state"]);m.cuda().eval().requires_grad_(False);state=snapshot(m)
  for sid in new_val:
   dest=C/f"{backbone}_seed{seed}_s{sid:02d}.npz";ap=dest.with_name(dest.stem+"_audit.json")
   if dest.exists() and ap.exists():
    a=json.loads(ap.read_text());assert sha(dest)==a["sha256"];feature_audits.append(a);continue
   x,y=(cbr_prepared(sid) if backbone=="cbramod" else lab_prepared(sid))
   with torch.inference_mode():
    logits=torch.cat([m(b.cuda()).cpu() for b in x.split(128)]);centered=(logits-logits.mean(1,keepdim=True)).numpy();prefix=m(x[:17].cuda()).cpu();prefix=(prefix-prefix.mean(1,keepdim=True)).numpy()
   err=float(abs(centered[:17]-prefix).max());assert err<1e-4
   np.savez_compressed(dest,logits=centered,y=y);a=dict(backbone=backbone,seed=seed,subject=sid,sha256=sha(dest),parent_sha256=sha(ip),prefix_logit_error=err);ap.write_text(json.dumps(a,indent=2));feature_audits.append(a)
  assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items());del m;torch.cuda.empty_cache();print("CONTROLLER LOGITS",backbone,seed,flush=True)
 (O/"feature_audit.json").write_text(json.dumps(dict(status="PASS",records=feature_audits,parent_states_exact=True),indent=2));records=[]
 for backbone,phase in phases.items():
  for seed in seeds:
   data={}
   for sid in train+original_val+new_val:
    path=C/f"{backbone}_seed{seed}_s{sid:02d}.npz" if sid in new_val else ROOT/phase/"logits"/f"seed{seed}_s{sid:02d}.npz";a=json.loads(path.with_name(path.stem+"_audit.json").read_text());assert sha(path)==a["sha256"];data[sid]=dict(np.load(path))
   for variant in variants:
    out=O/f"{backbone}_{variant}_seed{seed}";out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():records.append(json.loads((out/"complete.json").read_text()));continue
    orig=ROOT/phase/f"{variant}_seed{seed}";old=json.loads((orig/"complete.json").read_text());oldselected=torch.load(orig/"model.pt",map_location="cpu",weights_only=False);oldfinal=torch.load(orig/"final30.pt",map_location="cpu",weights_only=False)
    xx=np.concatenate([transform(data[s]["logits"],variant)[data[s]["y"]>=0] for s in train]);yy=np.concatenate([data[s]["y"][data[s]["y"]>=0] for s in train]);seed_all(seed);x=torch.from_numpy(xx);y=torch.from_numpy(yy);m=ResidualLogitHead(x.shape[1]);initial_hash=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest();assert initial_hash==old["initial_parameters_sha256"];m.mean.copy_(x.mean(0));m.scale.copy_(x.std(0).clamp(min=.1));m.cuda();x=x.cuda();y=y.cuda();evaldata={s:(torch.from_numpy(transform(data[s]["logits"],variant)).cuda(),data[s]["y"]) for s in original_val+new_val}
    def evaluate(sids):return {s:100*metrics(evaldata[s][1],hp(m,evaldata[s][0]))["macro_f1"] for s in sids}
    def compare_state(reference):return max(float(abs(v.cpu()-reference[k]).max()) for k,v in m.state_dict().items())
    def selected_replay():
     e=compare_state(oldselected["state"]);assert e<=5e-5;errors=[]
     for s in original_val:
      q=hp(m,evaldata[s][0]);gold=np.load(orig/f"s{s:02d}.npz")["p"];er=float(abs(q-gold).max());assert er<=2e-5 and np.array_equal(q.argmax(1),gold.argmax(1));errors.append(er)
     return dict(parameter_error=e,max_probability_error=max(errors),argmax_exact=True)
    original0=evaluate(original_val);base=evaluate(new_val);bn=float(np.mean(list(base.values())));history=[dict(epoch=0,mean_f1=bn,people=base,eligible=True,original_selector_people=original0)];best=bn;bestepoch=0;beststate=snapshot(m);reproduction=selected_replay() if old["selected_epoch"]==0 else None
    counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(seed);torch.manual_seed(100000+seed);torch.cuda.manual_seed_all(100000+seed);orders=[];old_metric_errors=[]
    for epoch in range(1,31):
     m.train();order=torch.randperm(len(y),generator=gen);orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[];start=time.perf_counter()
     for idx in order.split(128):
      idx=idx.cuda();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(x[idx]),y[idx],weight=weights);loss.backward();opt.step();losses.append(float(loss.detach()))
     oldscores=evaluate(original_val);scores=evaluate(new_val);mean=float(np.mean(list(scores.values())));eligible=mean>=bn-1 and all(scores[s]>=base[s]-2 for s in new_val);old_error=max(abs(oldscores[s]-old["history"][epoch]["people"][str(s)]) for s in original_val);old_metric_errors.append(old_error);history.append(dict(epoch=epoch,mean_f1=mean,people=scores,eligible=eligible,original_selector_people=oldscores,old_selector_metric_error=old_error,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
     if eligible and mean>best:best=mean;bestepoch=epoch;beststate=snapshot(m)
     if epoch==old["selected_epoch"]:reproduction=selected_replay()
    assert orders==old["order_hashes"] and reproduction is not None;finalerr=compare_state(oldfinal["state"]);assert finalerr<=5e-5
    ck=dict(state=beststate,dim=x.shape[1],variant=variant,seed=seed,epoch=bestepoch,parent_sha256=oldselected["parent_sha256"],source_selector=new_val);save(ck,out/"model.pt");m.load_state_dict(beststate);newreplay=evaluate(new_val);assert abs(np.mean(list(newreplay.values()))-best)<1e-10
    for sid,(vx,vy) in evaldata.items():np.savez_compressed(out/f"s{sid:02d}.npz",p=hp(m,vx),y=vy)
    rec=dict(backbone=backbone,seed=seed,variant=variant,selected_epoch=bestepoch,selected_f1=best,parent_f1=bn,people=newreplay,original_selector_evaluation=evaluate(original_val),old_selected_epoch=old["selected_epoch"],history=history,order_hashes=orders,initial_parameters_sha256=initial_hash,old_selected_reproduction=reproduction,old_final30_parameter_error=finalerr,max_old_selector_epoch_macro_f1_error=max(old_metric_errors),window_presentations=len(y)*30,checkpoint_sha256=sha(out/"model.pt"));(out/"complete.json").write_text(json.dumps(rec,indent=2));records.append(rec);del m,opt;torch.cuda.empty_cache();print("SELECTOR SWAP",backbone,seed,variant,bestepoch,best,"replay",finalerr,flush=True)
 (O/"selection.json").write_text(json.dumps(records,indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",models=len(records),initial_parameters_and_orders_exact=True,old_selected_checkpoint_compatibility=True,terminal_checkpoint_compatibility=True,new_selector_only=True,no_target_predictions=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()

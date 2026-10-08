from phase119_models import *
import hashlib,datetime,time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
O=ROOT/"phase125";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
@torch.inference_mode()
def probabilities(m,x):
 m.eval();return torch.cat([m(x[a:a+1024]).softmax(1).cpu() for a in range(0,len(x),1024)]).numpy()
def temporal_stratum(y):
 out=np.full(len(y),-1,dtype=np.int64)
 for t in range(10,len(y)):
  h=y[t-10:t+1]
  if (h>=0).all():out[t]=0 if np.all(h==h[-1]) else 1
 return out
def run():
 sources={"cbramod":"phase119","labram":"phase121"};train=[0,1,2,10,11,20,21,30,31,32];val=[5,15,25,35];seeds=[17,29,43];variants=["repeat11","history11"]
 pr=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose="Fixed source-only classconditional stable/transition balancing for static history heads; motivatedbyobservedcohortshift, no targetpriorfitting.",sources=sources,train=train,validation=val,seeds=seeds,variants=variants,weights="Keep original sqrtinverse currentclassweight. Within eachcurrenttrueclass, among windowswhosecurrent+past10labelsareknown, multiply by n_known_class/(2*n_class_stratum), where stratum is all11sameclass vs anychange. Ifonlyonestratum existsfactor1. Startup/unknownhistory factor1. Thusknown stable/transition have equal totalweight withinclass, original class-totalweight retained. No targetstatistics in weights.",loss="Weighted CE sum dividedbybatchsum of classweight*stratumfactor. Samefulltimeline features and exactscoredwindowcount, no labelcrop or synthetic labelsequence.",recipe="Originalphase119/121:zeroresidual MLP64/32,30epochs,AdamW.001wd.001,batch128,sourcefeaturemean/std floor.1,same per-seedorders/dropout RNG and 6021parametercapacity forrepeat/history.",selection="Earliestbest raw4sourceMacroF1 amongepoch0..30, meanwithin1pp/eachpersonwithin2pp of originalparent; epoch0fallback.",advance="Perbackbone onlyconsider oldtargetexploratoryreplication if3seed-mean withinperson weightedhistory beats originalhistory mean, improvesatleast3/4people againstoriginalhistory, and beats weightedrepeat mean; allsourceauditsPASS. Preservefailedbranches, no relaxingguard/retuningweights.",scope="Allsource/oldtargets previouslyobserved. Sourcepast/currentlabels determinetrainingweights; no true-labeltestgate. Additional30headtrainingepochs/branch, no newunique source labels. Fixedreplicationbothbackbones, no bestbackbone selection.",source_protocol_hashes={k:sha(ROOT/v/"protocol.json") for k,v in sources.items()},cpu_ids=list(range(8,24)),gpu=True)
 p=O/"protocol.json"
 if p.exists():
  old=json.loads(p.read_text());pr["created_utc"]=old["created_utc"];assert old==pr
 else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
 reference_labels={s:np.load(ROOT/"phase119/logits"/f"seed17_s{s:02d}.npz")["y"] for s in train};flat_y=np.concatenate([reference_labels[s][reference_labels[s]>=0] for s in train]);flat_stratum=np.concatenate([temporal_stratum(reference_labels[s])[reference_labels[s]>=0] for s in train]);factor=np.ones(len(flat_y),dtype=np.float32);factor_rows=[]
 for k in range(5):
  known=(flat_y==k)&(flat_stratum>=0);n=int(known.sum());counts=[int((known&(flat_stratum==j)).sum()) for j in [0,1]]
  for j in [0,1]:
   ix=known&(flat_stratum==j);f=n/(2*counts[j]) if min(counts)>0 else 1.;factor[ix]=f;factor_rows.append(dict(stage=k,stratum=j,n=counts[j],factor=f))
  assert abs(float(factor[known].sum())-n)<.01
 factors=torch.from_numpy(factor).cuda();(O/"weight_audit.json").write_text(json.dumps(dict(status="PASS",source_windows=len(flat_y),unknown_or_startup=int((flat_stratum<0).sum()),factor_min=float(factor.min()),factor_max=float(factor.max()),class_strata=factor_rows,known_class_weight_mass_preserved=True),indent=2));results=[]
 for backbone,phase in sources.items():
  for seed in seeds:
   data={}
   for sid in train+val:
    path=ROOT/phase/"logits"/f"seed{seed}_s{sid:02d}.npz";audit=json.loads(path.with_name(path.stem+"_audit.json").read_text());assert sha(path)==audit["sha256"];data[sid]=dict(np.load(path))
    if sid in train:assert np.array_equal(data[sid]["y"],reference_labels[sid])
   for variant in variants:
    out=O/f"{backbone}_{variant}_seed{seed}";out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():results.append(json.loads((out/"complete.json").read_text()));continue
    xx=np.concatenate([transform(data[s]["logits"],variant)[data[s]["y"]>=0] for s in train]);yy=np.concatenate([data[s]["y"][data[s]["y"]>=0] for s in train]);assert np.array_equal(yy,flat_y)
    seed_all(seed);m=ResidualLogitHead(xx.shape[1]);initial_hash=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest();x=torch.from_numpy(xx);m.mean.copy_(x.mean(0));m.scale.copy_(x.std(0).clamp(min=.1));m.cuda();x=x.cuda();y=torch.tensor(yy,dtype=torch.long,device="cuda");validation={s:(torch.from_numpy(transform(data[s]["logits"],variant)).cuda(),data[s]["y"]) for s in val}
    def evaluate():return {s:100*metrics(vy,probabilities(m,vx))["macro_f1"] for s,(vx,vy) in validation.items()}
    base=evaluate();bn=float(np.mean(list(base.values())));history=[dict(epoch=0,mean_f1=bn,people=base,eligible=True)];best=bn;bestepoch=0;beststate=snapshot(m);counts=torch.bincount(y,minlength=5).float();cw=(counts.sum()/counts.clamp(min=1)).sqrt();cw/=cw.mean();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(seed);torch.manual_seed(100000+seed);torch.cuda.manual_seed_all(100000+seed);orders=[]
    for epoch in range(1,31):
     m.train();order=torch.randperm(len(y),generator=gen);orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest());losses=[];start=time.perf_counter()
     for idx in order.split(128):
      idx=idx.cuda();opt.zero_grad(set_to_none=True);logits=m(x[idx]);ww=cw[y[idx]]*factors[idx];loss=(nn.functional.cross_entropy(logits,y[idx],reduction="none")*ww).sum()/ww.sum();loss.backward();assert torch.isfinite(loss);opt.step();losses.append(float(loss.detach()))
     scores=evaluate();mean=float(np.mean(list(scores.values())));eligible=mean>=bn-1 and all(scores[s]>=base[s]-2 for s in val);history.append(dict(epoch=epoch,mean_f1=mean,people=scores,eligible=eligible,loss=float(np.mean(losses)),seconds=time.perf_counter()-start))
     if eligible and mean>best:best=mean;bestepoch=epoch;beststate=snapshot(m)
    torch.save(dict(state=snapshot(m),dim=55,backbone=backbone,variant=variant,seed=seed,epoch=30),out/"final30.pt");torch.save(dict(state=beststate,dim=55,backbone=backbone,variant=variant,seed=seed,epoch=bestepoch),out/"model.pt");m.load_state_dict(beststate);replay=evaluate();assert abs(np.mean(list(replay.values()))-best)<1e-10
    for sid,(vx,vy) in validation.items():np.savez_compressed(out/f"s{sid:02d}.npz",p=probabilities(m,vx),y=vy)
    oldhist=json.loads((ROOT/phase/f"{variant}_seed{seed}"/"complete.json").read_text());assert orders==oldhist["order_hashes"] and initial_hash==oldhist["initial_parameters_sha256"]
    rec=dict(backbone=backbone,seed=seed,variant=variant,selected_epoch=bestepoch,selected_f1=best,parent_f1=bn,people=replay,history=history,initial_parameters_sha256=initial_hash,order_hashes=orders,window_presentations=len(y)*30,checkpoint_sha256=sha(out/"model.pt"),same_initial_parameters_and_orders_as_unweighted_recipe=True);(out/"complete.json").write_text(json.dumps(rec,indent=2));results.append(rec);del m,opt;torch.cuda.empty_cache();print("BALANCED",backbone,seed,variant,bestepoch,best,flush=True)
 (O/"selection.json").write_text(json.dumps(results,indent=2));(O/"audit.json").write_text(json.dumps(dict(status="PASS",models=len(results),same_window_presentations=True,training_weights_source_only=True,source_label_strata_not_input=True,same_initialization_and_orders_as_original=True),indent=2));print("COMPLETE",flush=True)
if __name__=="__main__":run()

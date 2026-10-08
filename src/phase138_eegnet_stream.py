
from phase86_meta_waveform import *
from phase27_train import FullCarrier,preprocess as labprep
from phase102_stream import CausalInputState
from phase127_stream import CausalLogitReadout
from phase2_temporal import ema
import time,datetime
O=ROOT/"phase138";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 specs=json.loads((ROOT/"phase79/protocol.json").read_text())
 reference=np.load(ROOT/"phase99/source_reference.npz")["reference"]
 cases=[c for c in json.loads((ROOT/"phase85/protocol.json").read_text())["cases"] if (c["domain"]=="source" and c["subject"]=="s05") or (c["subject"] in ["sub-III001","sub-I021"] and c["montage"]=="left")]
 assert len(cases)==3,[(c["domain"],c["subject"]) for c in cases]
 jobs=[dict(case=c,mode=mode) for c in cases for mode in (["raw"] if c["domain"]=="source" else ["raw","full"])]
 parents=[dict(s,model="eegnet") for s in json.loads((ROOT/"phase85/protocol.json").read_text())["models"]]
 protocol=dict(cases=jobs,parents=parents,seeds=[17,29,43],heads="Original source selected phase135 EEGNet-style history11; CPU and GPU parent implementations",ema=.5,unit="One already completed normalized two-lead 30second float32 2x3000 window; starts after EDF channel subtraction/resampling/centering/scaling. Not rawEDF acquisition validation.",checks="Full records, one-window normalization/filtering/encoder/readout/EMA; cached batch gold only for comparison. Mid-record joint JSON resume and reset first16 steps bitwise. No labels or target updates.",tolerance_probability=2e-5,scope="Three fixed records chosen by first identifier within source, healthy and oldpatient cohorts, no outcome selection. Timing includes waveform preprocessing, CPU or GPU parent and CPU readout; excludes disk, cold start, acquisition and model loading.",implementation_hashes={name:sha(ROOT/"src"/name) for name in ["phase102_stream.py","phase127_stream.py","phase138_eegnet_stream.py"]})
 pp=O/"protocol.json"
 if pp.exists():assert json.loads(pp.read_text())==protocol
 else:pp.write_text(json.dumps(protocol,indent=2));(O/"protocol.sha256").write_text(sha(pp))
 records=[]
 for back in ["eegnet_cpu","eegnet_gpu"]:
  device="cpu" if back.endswith("cpu") else "cuda"
  cachephase=136;headphase=135
  for sp in parents:
   seed=sp["seed"];assert sha(sp["path"])==sp["sha256"]
   m,_=load_model(sp["path"])
   m.load_state_dict(torch.load(sp["path"],map_location="cpu",weights_only=False)["state"]);m.to(device).eval().requires_grad_(False);initial=snapshot(m)
   hp=ROOT/f"phase{headphase}"/f"history11_seed{seed}"/"model.pt"
   for job in jobs:
    c=job["case"];mode=job["mode"];name="_".join(c[k] for k in ["domain","subject","montage"])
    dest=O/f"{back}_seed{seed}_{name}_{mode}.json"
    if dest.exists():
     records.append(json.loads(dest.read_text()));continue
    raw=np.load(c["x"],mmap_mode="r");cp=ROOT/f"phase{cachephase}"/"predictions"/f"{name}_{mode}.npz";gold=np.load(cp)
    gp=gold[f"history11_seed{seed}"];gq=ema(gp,.5);parentgold=gold[f"frozen_seed{seed}"]
    def new():
     normal=CausalInputState(specs["source_reference_rms"],specs["lower_source_rms_quantile"],reference,"full") if mode=="full" else None
     read=CausalLogitReadout(hp,sp["sha256"],f"{back}:normalized30s:{mode}",.5)
     return normal,read
    normal,read=new()
    def step(i,normal,read):
     w=np.array(raw[i],copy=True)
     if normal is not None:w,_=normal.step(w)
     xx=torch.from_numpy(w[None])
     with torch.inference_mode():
      z=m(xx.to(device));z=(z-z.mean(1,keepdim=True)).cpu();parent=z.softmax(1)[0].numpy()
     p,q=read.step(z[0].numpy())
     return parent,p,q
    maxerr=np.zeros(3);argdiff=np.zeros(3,dtype=int);timings=[];first=[];resume_exact=True
    for i in range(len(raw)):
     if i==len(raw)//2:
      bundle=json.loads(json.dumps(dict(normal=None if normal is None else normal.export_state(),read=read.export_state()),allow_nan=False))
      n2,r2=new()
      if n2 is not None:n2.restore_state(bundle["normal"])
      r2.restore_state(bundle["read"])
      for k in range(i,min(i+16,len(raw))):
       a=step(k,normal,read);b=step(k,n2,r2)
       resume_exact=resume_exact and all(np.array_equal(x,y) for x,y in zip(a,b))
      assert resume_exact
      normal,read=new()
      if normal is not None:normal.restore_state(bundle["normal"])
      read.restore_state(bundle["read"])
     start=time.perf_counter();got=step(i,normal,read);elapsed=time.perf_counter()-start
     if i>=16:timings.append(elapsed*1000)
     refs=(parentgold[i],gp[i],gq[i])
     maxerr=np.maximum(maxerr,[float(abs(a-b).max()) for a,b in zip(got,refs)])
     argdiff+=np.array([a.argmax()!=b.argmax() for a,b in zip(got,refs)],int)
     if i<16:first.append(tuple(v.copy() for v in got))
    assert max(maxerr)<protocol["tolerance_probability"],(back,seed,name,mode,maxerr)
    normal,read=new()
    reset_exact=all(all(np.array_equal(a,b) for a,b in zip(step(i,normal,read),ref)) for i,ref in enumerate(first))
    assert reset_exact
    assert all(torch.equal(v.cpu(),initial[k]) for k,v in m.state_dict().items())
    rec=dict(backbone=back,device=device,parent_parameters=sum(p.numel() for p in m.parameters()),head_parameters=sum(p.numel() for p in read._model.parameters()),seed=seed,domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,windows=len(raw),parent_probability_error=maxerr[0],history_probability_error=maxerr[1],ema_probability_error=maxerr[2],parent_argmax_differences=int(argdiff[0]),history_argmax_differences=int(argdiff[1]),ema_argmax_differences=int(argdiff[2]),joint_json_resume_exact=resume_exact,reset_exact=reset_exact,parent_parameters_unchanged=True,median_ms=float(np.median(timings)),p95_ms=float(np.quantile(timings,.95)),timed_windows=len(timings),gold_sha256=sha(cp),head_sha256=sha(hp))
    dest.write_text(json.dumps(rec,indent=2));records.append(rec);print("STREAM",back,seed,name,mode,len(raw),maxerr.tolist(),argdiff.tolist(),flush=True)
   del m;torch.cuda.empty_cache()
 assert len(records)==30
 audit=dict(status="PASS",streams=len(records),windows=sum(r["windows"] for r in records),labels_not_read=True,no_updates=True,joint_resume_exact=all(r["joint_json_resume_exact"] for r in records),reset_exact=all(r["reset_exact"] for r in records),max_parent_error=max(r["parent_probability_error"] for r in records),max_history_error=max(r["history_probability_error"] for r in records),max_ema_error=max(r["ema_probability_error"] for r in records),total_argmax_differences=sum(r[k] for r in records for k in ["parent_argmax_differences","history_argmax_differences","ema_argmax_differences"]),records=records)
 (O/"audit.json").write_text(json.dumps(audit,indent=2));print("COMPLETE",json.dumps({k:v for k,v in audit.items() if k!="records"}),flush=True)
if __name__=="__main__":
 with sdpa_kernel(SDPBackend.MATH):run()

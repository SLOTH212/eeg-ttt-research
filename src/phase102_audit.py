from pathlib import Path
import json,time,hashlib,numpy as np
from phase102_stream import CausalInputState
from phase79_amplitude import normalize
from phase99_covariance import transform
R=Path(__file__).resolve().parents[1];O=R/"phase102";O.mkdir(exist_ok=True);pr=json.loads((R/"phase79/protocol.json").read_text());reference=np.load(R/"phase99/source_reference.npz")["reference"];cases=json.loads((R/"phase85/protocol.json").read_text())["cases"]
def new(mode):return CausalInputState(pr["source_reference_rms"],pr["lower_source_rms_quantile"],reference,mode)
rows=[];start=time.perf_counter()
for c in cases:
 raw=np.load(c["x"],mmap_mode="r");scalar,gain,logs,_=normalize(raw,pr["source_reference_rms"],"guarded_median")
 for mode in ["scalar","diagonal","full"]:
  expected=scalar if mode=="scalar" else transform(raw,gain,reference,mode)[0];state=new(mode);error=0.;exact=True
  for i,x in enumerate(raw):
   if i==min(257,len(raw)//2):
    serialized=json.dumps(state.export_state(),allow_nan=False);restored=new(mode);restored.restore_state(json.loads(serialized));state=restored
   out,info=state.step(x);error=max(error,float(abs(out-expected[i]).max()));exact=exact and np.array_equal(out,expected[i]);assert abs(info["gain"]-gain[i])<1e-12
  assert error<2e-6
  state.reset();first,_=state.step(raw[0]);assert np.array_equal(first,expected[0]);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,windows=len(raw),max_error=error,bitwise_exact=bool(exact),serialized_resume=True,reset_exact=True))
 print("STREAM",c["domain"],c["subject"],c["montage"],flush=True)
# Leading and interspersed flat windows and rank-one channels are legitimate states.
rng=np.random.default_rng(102017);raw=rng.normal(0,.01,size=(16,2,3000)).astype(np.float32);raw[:3]=0;raw[8]=0;raw[12,1]=raw[12,0]
scalar,gain,_,_=normalize(raw,pr["source_reference_rms"],"guarded_median")
for mode in ["scalar","floor","diagonal","full"]:
 expected=normalize(raw,pr["source_reference_rms"],"floor")[0] if mode=="floor" else scalar if mode=="scalar" else transform(raw,gain,reference,mode)[0];state=new(mode)
 for i,x in enumerate(raw):
  state.restore_state(json.loads(json.dumps(state.export_state(),allow_nan=False)));out,_=state.step(x);assert np.max(abs(out-expected[i]))<2e-6
 state.reset();assert state.export_state()["log_rms"] is None and state.export_state()["shape"] is None
probe=new("full"); exported=probe.export_state(); exported["config"]["reference_shape"][0][0]+=1; assert probe.config["reference_shape"][0][0]==reference[0,0]
bad=new("full").export_state();bad["config"]["mode"]="diagonal"
try:new("full").restore_state(bad);raise AssertionError("configuration mismatch accepted")
except ValueError:pass
audit=dict(status="PASS",records=rows,cases=len(cases),streams=len(rows),windows=sum(v["windows"] for v in rows),seconds=time.perf_counter()-start,max_error=max(v["max_error"] for v in rows),all_bitwise_exact=all(v["bitwise_exact"] for v in rows),flat_singular_resume_tests=True,configuration_mismatch_rejected=True,bounded_state="One scalar logRMS, 2x2 shape EMA, counter and fixed source constants; no full record, labels, or future windows.",implementation_sha256=hashlib.sha256((R/"src/phase102_stream.py").read_bytes()).hexdigest())
(O/"audit.json").write_text(json.dumps(audit,indent=2));print({k:v for k,v in audit.items() if k!="records"})
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:35px auto}</style><h1>逐窗归一化及断点恢复验证</h1><p>每次只接收一个已完成的30秒窗口，保持标量幅值与二维形状的因果状态。状态可序列化、恢复和按新记录重置。模型分类与输入预处理仍遵守先前方法，本阶段不新增方法或调整阈值。</p><pre>'+json.dumps({k:v for k,v in audit.items() if k!="records"},indent=2)+'</pre><p>在先前92条记录上与离线批处理对照，另含平坦窗口、秩一通道、逐步保存恢复及配置不匹配拒绝。此处只验证输入实现，不声称分类器完整在线服务已经部署。</p>')

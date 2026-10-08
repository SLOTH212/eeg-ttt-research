from phase127_stream import *
from common import ROOT
from phase2_temporal import ema
import time,os
os.environ["CUDA_VISIBLE_DEVICES"]=""
O=ROOT/"phase127";O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pr=dict(purpose="Actual one-window frozen temporalreadout state implementation, evaluated using previouslycached centeredparentlogits; no waveformencoder recomputation.",backbones={"cbramod":{"heads":"phase119","predictions":"phase120"},"labram":{"heads":"phase121","predictions":"phase123"}},primary_ema=.5,implementation_sha256=sha(ROOT/"src/phase127_stream.py"),bounded_state="Atmost10previous5float32logits (200bytes),5float32EMAvalues (20bytes),countandconfiguration metadata. No labels/fullrecord/futureinput.",scope="900case-input-model streams from150conditions x2backbones x3seeds. CPU readoutonly, notwholepipeline deploymentproof. Compareallraw/EMAoutputs to cachedGPU batchhead probabilities, recordargmaxdifferences; tolerance2e-5.",resume="Every stream JSONroundtrip atmidpoint; compare16resumedsteps bitwise to uninterruptedCPUstream, then continuefromrestoredinstance. Reset first3windows replaybitwise. Input and exportedstate aliasing,configmismatch and invalidstate checks.",timing="Perstep CPUreadout timings for firstfixedsource_s05/raw/seed17/cbramod stream; excludesparentencoder/inputpreprocessing/load/serialization. No energy orhardwaregeneralization.",cpu_ids=list(range(8,24)),gpu=False)
p=O/"protocol.json"
if p.exists():assert json.loads(p.read_text())==pr
else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
records=[];timings=[];total=0
for backbone,paths in pr["backbones"].items():
 audit=json.loads((ROOT/paths["predictions"]/"audit.json").read_text());assert audit["status"]=="PASS"
 for rec in audit["records"]:
  c=rec["case"];name="_".join(c[k] for k in ["domain","subject","montage"]);mode=rec["mode"];z=np.load(ROOT/paths["predictions"]/"predictions"/f"{name}_{mode}.npz")
  for seed in [17,29,43]:
   headpath=ROOT/paths["heads"]/f"history11_seed{seed}"/"model.pt";ck=torch.load(headpath,map_location="cpu",weights_only=False);parent=ck["parent_sha256"];tag=backbone+"/"+mode;runtime=CausalLogitReadout(headpath,parent,tag,.5);initial={k:v.clone() for k,v in runtime._model.state_dict().items()};x=z[f"logits_seed{seed}"];expected=z[f"history11_seed{seed}"];expected_ema=ema(expected,.5);maxraw=0.;maxema=0.;argraw=0;argema=0;first=[];resumed=None;mid=len(x)//2
   for i,logit in enumerate(x):
    measure=backbone=="cbramod" and name=="source_s05_native" and mode=="raw" and seed==17
    if measure:start=time.perf_counter_ns()
    raw,smoothed=runtime.step(logit)
    if measure and i>=10:timings.append((time.perf_counter_ns()-start)/1e6)
    if i<3:first.append((raw.copy(),smoothed.copy()))
    maxraw=max(maxraw,float(abs(raw-expected[i]).max()));maxema=max(maxema,float(abs(smoothed-expected_ema[i]).max()));argraw+=int(raw.argmax()!=expected[i].argmax());argema+=int(smoothed.argmax()!=expected_ema[i].argmax())
    if i==mid:
     state=json.loads(json.dumps(runtime.export_state()));resumed=CausalLogitReadout(headpath,parent,tag,.5);resumed.restore_state(state)
    elif mid<i<=mid+16:
     r2,s2=resumed.step(logit);assert np.array_equal(r2,raw) and np.array_equal(s2,smoothed)
     if i==mid+16:runtime=resumed
   assert maxraw<2e-5 and maxema<2e-5
   saved=runtime.export_state();assert len(saved["history"])==10 and saved["count"]==len(x);saved["history"][0][0]+=100;assert runtime.export_state()["history"][0][0]!=saved["history"][0][0]
   assert all(torch.equal(v,initial[k]) for k,v in runtime._model.state_dict().items());runtime.reset()
   for i in range(3):
    rp,sp=runtime.step(x[i]);assert np.array_equal(rp,first[i][0]) and np.array_equal(sp,first[i][1])
   records.append(dict(backbone=backbone,case=name,mode=mode,seed=seed,windows=len(x),max_raw_probability_error=maxraw,max_ema_probability_error=maxema,raw_argmax_differences=argraw,ema_argmax_differences=argema,resume_bitwise_exact=True,reset_bitwise_exact=True,weights_frozen_exact=True));total+=len(x)
  print("STREAM",backbone,name,mode,flush=True)
# Fixed configuration and alias audits without labels.
paths=pr["backbones"]["cbramod"];headpath=ROOT/paths["heads"]/"history11_seed17/model.pt";parent=torch.load(headpath,map_location="cpu",weights_only=False)["parent_sha256"];z=np.load(ROOT/"phase120/predictions/source_s05_native_raw.npz");x=z["logits_seed17"];headp=z["history11_seed17"]
for alpha in [1.,.8]:
 rr=CausalLogitReadout(headpath,parent,"cbramod/raw",alpha);got=np.array([rr.step(v)[1] for v in x[:65]]);assert abs(got-ema(headp[:65],alpha)).max()<2e-5
a=CausalLogitReadout(headpath,parent,"cbramod/raw",.5);b=CausalLogitReadout(headpath,parent,"cbramod/raw",.5);v=x[0].copy();a.step(v);b.step(x[0]);v[:]=0
assert all(np.array_equal(aa,bb) for aa,bb in zip(a.step(x[1]),b.step(x[1])))
state=a.export_state();wrong=CausalLogitReadout(headpath,parent,"cbramod/full",.5)
try:wrong.restore_state(state);raise AssertionError("Configuration mismatch accepted")
except ValueError:pass
bad=json.loads(json.dumps(state));bad["count"]+=1
try:a.restore_state(bad);raise AssertionError("Invalid count/history accepted")
except ValueError:pass
before=a.export_state()
try:a.step(np.full(5,np.nan,dtype=np.float32));raise AssertionError("Nonfinite input accepted")
except ValueError:pass
assert a.export_state()==before
assert len(records)==900
audit=dict(status="PASS",streams=len(records),window_steps=total,max_raw_probability_error=max(r["max_raw_probability_error"] for r in records),max_ema_probability_error=max(r["max_ema_probability_error"] for r in records),raw_argmax_differences=sum(r["raw_argmax_differences"] for r in records),ema_argmax_differences=sum(r["ema_argmax_differences"] for r in records),resume_reset_bitwise_exact=True,state_input_alias_checks=True,configuration_rejection=True,invalid_input_atomic=True,model_weights_unchanged_exact=True,readout_only_cpu_timing_ms=dict(n=len(timings),median=float(np.median(timings)),p95=float(np.quantile(timings,.95))),records=records)
(O/"audit.json").write_text(json.dumps(audit,indent=2));(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1150px;margin:35px auto}</style><h1>逐窗口静态历史读取器：状态与恢复审计</h1><p>新增CausalLogitReadout，每次只接收当前5类已中心化基础模型logit，保存最多10个过去向量及固定EMA状态。首次边缘填充、逐窗EMA与既有离线实现一致。模型权重固定，状态绑定头/父模型SHA256、输入处理标识和EMA系数；配置不匹配时拒绝恢复。</p><p>两种基础模型、三个种子、150个记录-输入条件，共900条流。这里复用缓存的父模型logit，在CPU上实际逐窗运行读取器；不是整条波形编码管线的重新部署验证。每条流中途JSON序列化恢复后，与未中断流比较16步逐元素一致，再继续恢复后的流；重置后前3窗也逐元素一致。</p><pre>'+json.dumps({k:v for k,v in audit.items() if k!="records"},indent=2)+'</pre><p>与原GPU批量头比较允许小幅浮点差异，并明确记录分类差异；没有标签读取或模型更新。计时仅为读取器步骤，排除波形预处理、父编码器、载入和序列化。数组数据载荷最多220字节，不含Python对象、计数、配置元数据及模型权重。<a href="../src/phase127_stream.py">读取器实现</a> · <a href="protocol.json">固定审计范围</a> · <a href="audit.json">逐流审计</a></p>');print({k:v for k,v in audit.items() if k!="records"},flush=True)

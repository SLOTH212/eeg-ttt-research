from phase86_meta_waveform import *
from phase77_predict import SpectralNet
from phase76_prepare import features
import contextlib,platform,sys,pandas as pd
O=ROOT/"phase113";O.mkdir(exist_ok=True)
assert json.loads((ROOT/"phase112/source_replay_audit.json").read_text())["status"]=="PASS"
methods=["teacher_cpu","teacher_gpu","cbramod_default","cbramod_math","teacher_write_math"];times=np.linspace(10,2721,64,dtype=int)
pr=dict(purpose="Warm batch1 online processing cost, not new accuracy experiment; staticCPU/GPUteacher versusCBraMod andfixedteacherWRITE.",source_subject=5,times=times.tolist(),methods=methods,warmups_per_method=10,measured_windows_per_method=64,order="Randomized methodorder within eachwindow, RNG113017; no concurrentprojectGPUtraining duringbenchmark.",input="Already100Hz centered/scaled/clipped 2x3000 signal, RAM resident. Disk/EDFreading and acquisitionwait excluded. Include currentCPUWelch forcontextteacher and CPUbandpass/upsampling forCBr.",teacher_history="Previous10cachedfeaturevectors represent retainedonlinehistory; currentfeature recomputed inside everytimedteacherprediction.",write="Predictcurrentstudent first, then currentteacher softCE, LN9600 gradientclip1/lr.01. No futureinput or trueclass. Freshrestore outside timing for repeatable pulse cost, notcontinuousaccuracy evaluation.",backend="FP32,TF32off; default frozenCBr uses defaultSDPA/MHAfastpath; matchedmath frozen andWRITE use MATH/fastpathoff asphase104. Teacheris oldcontext11seed17.",timing="Wallclock median/p95; predictionready separatedfromupdatecomplete. GPU output synchronized; incremental peakallocatedGPUmemory abovealreadyresidentmodels, notwholemachineRAM.",scope="One hardware/softwareconfiguration with warmmodels; no energy/clinical/throughput extrapolation.",cpu_ids=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),python=sys.version.split()[0])
assert pr["cpu_ids"]==list(range(8,24));(O/"protocol.json").write_text(json.dumps(pr,indent=2))
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
path=ROOT/"phase4/results/context11_mixed_17/model.pt";ck=torch.load(path,map_location="cpu",weights_only=False);teacher_cpu=SpectralNet(ck["dim"]);teacher_cpu.load_state_dict(ck["state"]);teacher_cpu.eval().requires_grad_(False);teacher_gpu=SpectralNet(ck["dim"]);teacher_gpu.load_state_dict(ck["state"]);teacher_gpu.cuda().eval().requires_grad_(False);teacher_snap=snapshot(teacher_gpu)
m=meta_model();m.eval();params=[p for n,p in ln_named(m)];initial=[p.detach().clone() for p in params];model_snap=snapshot(m)
raw=np.load(ROOT/"data/processed/s05_x.npy",mmap_mode="r");ff=np.load(ROOT/"data/processed/s05_meta.npz")["features"];cases=[(int(t),np.array(raw[t]),np.array(ff[t-10:t])) for t in times]
def restore():
 with torch.no_grad():
  for p,v in zip(params,initial):p.copy_(v)
def teacher_input(window,past):
 current=features(window[None]);return torch.from_numpy(np.concatenate([past,current]).reshape(1,-1))
def execute(method,window,past):
 torch.backends.mha.set_fastpath_enabled(method=="cbramod_default");math=method in ["cbramod_math","teacher_write_math"];backend=sdpa_kernel(SDPBackend.MATH) if math else contextlib.nullcontext()
 with backend:
  torch.cuda.synchronize();base_alloc=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  if method.startswith("teacher_") and method!="teacher_write_math":
   tx=teacher_input(window,past)
   with torch.inference_mode():p=(teacher_cpu(tx) if method=="teacher_cpu" else teacher_gpu(tx.cuda())).softmax(1).cpu().numpy()
   ready=time.perf_counter()
  else:
   x=torch.from_numpy(prep(window[None],[.1,45])).cuda()
   if method!="teacher_write_math":
    with torch.inference_mode():p=m(x).softmax(1).cpu().numpy()
    ready=time.perf_counter()
   else:
    logits=m(x);p=logits.softmax(1).detach().cpu().numpy();ready=time.perf_counter();tx=teacher_input(window,past)
    with torch.no_grad():q=teacher_gpu(tx.cuda()).double().log_softmax(1).exp()
    loss=-(q*logits.double().log_softmax(1)).sum();grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));mult=.01*torch.clamp(1/(norm+1e-6),max=1.)
    with torch.no_grad():
     for par,v,g in zip(params,initial,grads):par.copy_(v-mult*g)
  if method!="teacher_cpu":torch.cuda.synchronize()
  end=time.perf_counter();extra=max(0,torch.cuda.max_memory_allocated()-base_alloc)
 return p,dict(prediction_ready_ms=1000*(ready-start),complete_ms=1000*(end-start),extra_peak_gpu_mib=extra/(1024**2))
for method in methods:
 for j in range(10):
  restore();_,window,past=cases[j];execute(method,window,past)
restore();rows=[];checks=[];rng=np.random.default_rng(113017)
for t,window,past in cases:
 feat_error=float(abs(features(window[None])[0]-ff[t]).max());assert feat_error<2e-5;probs={}
 for method in rng.permutation(methods):
  restore();p,rec=execute(method,window,past);assert np.isfinite(p).all();probs[method]=p.copy();rows.append(dict(t=t,method=str(method),**rec))
 cpu_gpu=float(abs(probs["teacher_cpu"]-probs["teacher_gpu"]).max());backend_error=float(abs(probs["cbramod_default"]-probs["cbramod_math"]).max());write_current_error=float(abs(probs["teacher_write_math"]-probs["cbramod_math"]).max());assert max(cpu_gpu,backend_error,write_current_error)<2e-5
 checks.append(dict(t=t,feature_cache_error=feat_error,teacher_cpu_gpu_error=cpu_gpu,student_backend_error=backend_error,current_prediction_before_write_error=write_current_error))
restore();assert all(torch.equal(v.cpu(),model_snap[k]) for k,v in m.state_dict().items()) and all(torch.equal(v.cpu(),teacher_snap[k]) for k,v in teacher_gpu.state_dict().items())
df=pd.DataFrame(rows);df.to_csv(O/"timings.csv",index=False);summary=df.groupby("method").agg(windows=("t","size"),prediction_median_ms=("prediction_ready_ms","median"),prediction_p95_ms=("prediction_ready_ms",lambda x:float(np.quantile(x,.95))),complete_median_ms=("complete_ms","median"),complete_p95_ms=("complete_ms",lambda x:float(np.quantile(x,.95))),incremental_peak_gpu_mib=("extra_peak_gpu_mib","max")).reset_index();summary.to_csv(O/"summary.csv",index=False)
audit=dict(status="PASS",timed_windows=len(rows),checks=checks,model_state_restored_exact=True,teacher_state_frozen_exact=True,true_labels_not_read=True,tf32_disabled=True,cpu_affinity=sorted(os.sched_getaffinity(0)))
(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>单窗口在线处理成本：静态教师与参数WRITE</h1><p>模型预热后，64个固定源窗口，每窗随机排列五个方法。当前信号已在RAM且为既有100Hz标准输入；教师当前窗Welch与CBraMod滤波/升采样计入时间，过去10窗特征视为在线已保留状态。磁盘读取、设备采集等待和冷启动不计入。CPU限定8–23核，Torch两线程，GPU '+pr["gpu"]+'，FP32/TF32关闭。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>prediction表示当前预测就绪；complete包括教师soft损失的9600个LN参数梯度、裁剪和写入。每次测试前在计时外重置，用于比较单次操作成本，不代表持续更新的准确率。增量GPU峰值只统计已驻留模型之上的分配，不能理解成整个进程或机器内存。默认静态推理与匹配MATH后端分别报告，避免把后端差异混入算法结论。</p><p>真实标签未读取。CPU/GPU教师、不同学生后端及WRITE前当前预测均通过数值兼容检查，全部模型状态最终恢复。仅代表此硬件、软件和温热状态；不是能耗测量、临床实时性认证或其他机器性能保证。</p><p><a href="protocol.json">硬件与固定计时范围</a> · <a href="audit.json">数值与状态核对</a> · <a href="timings.csv">逐次时延</a> · <a href="../phase104/report.html">既有分类效用比较</a></p>')
print(summary.to_string(index=False));print({k:v for k,v in audit.items() if k!="checks"})

from pathlib import Path
import json,hashlib,os
import numpy as np,pandas as pd,pyedflib
from scipy.signal import resample_poly
from phase76_prepare import header,direct_samples,transform
R=Path(__file__).resolve().parents[1];O=R/"phase98";O.mkdir(exist_ok=True);assert os.sched_getaffinity(0)<=set(range(8,24))
old=json.loads((R/"phase76/protocol.json").read_text());names=old["native_channels"];pr=dict(purpose="Read-only physical amplitude and montage-cancellation audit of existing10healthy ISRUC records; no modelorlabel selection.",subjects=old["participants"],native_channels=names,windows="64equallyspaced full30s windows/person from signal length only",comparison="Native referenced EEG amplitudes versus locked bipolar differences, both200Hz physicalunits and actual100Hzinput; no claim of exact montageequivalence.",labels_read=False,gpu=False,source_reference_rms_uv=100*json.loads((R/"phase79/protocol.json").read_text())["source_reference_rms"])
(O/"protocol.json").write_text(json.dumps(pr,indent=2));rows=[];checks=[]
for sid in pr["subjects"]:
 path=R/"phase76/raw"/sid/"eeg"/(sid+"_task-sleep_eeg.edf");expected=old["raw_files_sha256"][str(path.relative_to(R/"phase76/raw"))];assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;h=header(path);left=np.load(R/"phase76/processed"/(sid+"_left_x.npy"),mmap_mode="r");right=np.load(R/"phase76/processed"/(sid+"_right_x.npy"),mmap_mode="r");times=np.linspace(0,len(left)-1,64,dtype=int);identityerr=0.;physicalerr=0.
 with pyedflib.EdfReader(str(path),annotations_mode=0) as f:
  idx=[f.getSignalLabels().index(n) for n in names];assert all(f.getSampleFrequency(i)==200 for i in idx)
  for n,i in zip(names,idx):
   hi=h["labels"].index(n);assert h["units"][hi] in ["uV","µV"];physicalerr=max(physicalerr,float(abs(f.readSignal(i,start=0,n=1000)-direct_samples(path,h,hi)).max()))
  assert physicalerr<1e-8
  for t0 in times:
   t=int(t0);x=np.stack([f.readSignal(i,start=t*6000,n=6000) for i in idx]);cached=transform(x[None])[0];assert np.array_equal(cached[:2],left[t]) and np.array_equal(cached[2:],right[t]);center=x-x.mean(-1,keepdims=True);var=np.mean(center**2,axis=1);native=resample_poly(x,1,2,axis=-1).astype(np.float32);native-=native.mean(-1,keepdims=True);nrms=np.sqrt(np.mean(native.astype(float)**2,axis=1));rrms=np.sqrt(np.mean((cached.astype(float)*100)**2,axis=1))
   for j,name in enumerate(names):rows.append(dict(subject=sid,t=t,kind="native",channel=name,rms_uv=nrms[j],native200_rms_uv=np.sqrt(var[j]),pair_correlation=np.nan,difference_to_pair_rms=np.nan))
   for j,(a,b,name) in enumerate([(0,1,"F3-C3"),(1,2,"C3-O1"),(3,4,"F4-C4"),(4,5,"C4-O2")]):
    cov=float(np.mean(center[a]*center[b]));corr=cov/np.sqrt(var[a]*var[b]) if var[a]*var[b]>0 else np.nan;difference=np.mean((center[a]-center[b])**2);identityerr=max(identityerr,abs(difference-(var[a]+var[b]-2*cov)));ratio=np.sqrt(difference/(.5*(var[a]+var[b]))) if var[a]+var[b]>0 else np.nan;rows.append(dict(subject=sid,t=t,kind="locked_bipolar",channel=name,rms_uv=rrms[j],native200_rms_uv=np.sqrt(difference),pair_correlation=corr,difference_to_pair_rms=ratio))
   for side,inds,bip in [("left",(0,2),(0,1)),("right",(3,5),(2,3))]:
    rows.append(dict(subject=sid,t=t,kind="joint_native_front_posterior",channel=side,rms_uv=float(np.sqrt(np.mean(nrms[list(inds)]**2))),native200_rms_uv=np.nan,pair_correlation=np.nan,difference_to_pair_rms=np.nan))
    rows.append(dict(subject=sid,t=t,kind="joint_locked_bipolar",channel=side,rms_uv=float(np.sqrt(np.mean(rrms[list(bip)]**2))),native200_rms_uv=np.nan,pair_correlation=np.nan,difference_to_pair_rms=np.nan))
 assert identityerr<1e-8;checks.append(dict(subject=sid,windows=64,times=times.tolist(),raw_sha256=expected,physical_calibration_error=physicalerr,cached_bipolar_inputs_exact=True,variance_identity_error=identityerr));print("MONTAGE AUDIT",sid,flush=True)
df=pd.DataFrame(rows);df.to_csv(O/"window_amplitudes.csv",index=False);person=df.groupby(["subject","kind","channel"])[["rms_uv","pair_correlation","difference_to_pair_rms"]].median().reset_index();person.to_csv(O/"person_amplitudes.csv",index=False);summary=person.groupby(["kind","channel"])[["rms_uv","pair_correlation","difference_to_pair_rms"]].median();summary.to_csv(O/"summary.csv")
audit=dict(status="PASS",people=10,sampled_windows=640,records=checks,no_labels_read=True,no_gpu_used=True,no_signal_or_model_modified=True);(O/"audit.json").write_text(json.dumps(audit,indent=2));print(summary.to_string())
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>外部健康队列：原始参考导联与双极差分的幅值诊断</h1><p>只读现有原始EDF，每人按信号长度等距取64个完整窗口，共640个；不读取分期标签，不运行或改变模型。独立数字到物理单位校准、双极输入缓存逐元素一致性及差分方差恒等式均校验。该抽样用于解释幅值来源，不代表对全部8,889窗口重新统计。</p>'+summary.reset_index().to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>difference_to_pair_rms是差分通道RMS除以两个原通道均方RMS；它与相关性共同反映相减的幅值影响。不同参考和空间位置并不等同，即便数值幅值接近也不能据此宣称可互换。表中先取每人窗口中位数，再取10人中位数。</p><p><a href="audit.json">物理单位与差分审计</a> · <a href="person_amplitudes.csv">逐人结果</a> · <a href="protocol.json">固定只读诊断方案</a></p>')

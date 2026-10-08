from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,pyedflib
from scipy.io import loadmat
from scipy.signal import welch
R=Path(__file__).resolve().parents[1];O=R/"phase111";matpath=O/"scratch/subject1.mat";edf=R/"phase76/raw/sub-III001/eeg/sub-III001_task-sleep_eeg.edf"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
download=json.loads((O/"download_audit.json").read_text());assert download["status"]=="PASS" and sha(matpath)==download["plaintext_sha256"]
old=json.loads((R/"phase76/protocol.json").read_text());assert sha(edf)==old["raw_files_sha256"][str(edf.relative_to(R/"phase76/raw"))]
names=["F3_A2","C3_A2","O1_A2","F4_A1","C4_A1","O2_A1"];data=loadmat(matpath,variable_names=names);rows=[];windows=[];psds={};snippets={};bands=[]
with pyedflib.EdfReader(str(edf),annotations_mode=0) as f:
 labels=f.getSignalLabels()
 for name in names:
  label=name.replace("_","-");i=labels.index(label);assert f.getSampleFrequency(i)==200 and f.getPhysicalDimension(i)=="uV";y=data[name];assert y.dtype==np.float64 and y.ndim==2 and y.shape[1]==6000 and np.isfinite(y).all();n=len(y);total=int(f.getNSamples()[i]);assert total//6000-n==30 and total%6000==0
  x=f.readSignal(i,start=0,n=n*6000).reshape(n,6000);xc=x-x.mean(axis=1,keepdims=True);yc=y-y.mean(axis=1,keepdims=True);vx=np.mean(xc*xc,axis=1);vy=np.mean(yc*yc,axis=1);cross=np.mean(xc*yc,axis=1);valid=(vx>1e-20)&(vy>1e-20);ratios=np.sqrt(vy[valid]/vx[valid]);corr=cross[valid]/np.sqrt(vx[valid]*vy[valid]);slope=float(cross.sum()/vx.sum());rmsx=np.sqrt(np.mean(x*x,axis=1));rmsy=np.sqrt(np.mean(y*y,axis=1))
  rec=dict(channel=name,edf_declared_unit="uV",mat_unit_metadata="No explicit unit variable; numerical comparison only",overlap_epochs=n,edf_full_epochs=total//6000,omitted_epochs=30,centered_sample_correlation=float(cross.sum()/np.sqrt(vx.sum()*vy.sum())),centered_linear_slope_mat_over_edf=slope,window_std_ratio_q05=float(np.quantile(ratios,.05)),window_std_ratio_median=float(np.median(ratios)),window_std_ratio_q95=float(np.quantile(ratios,.95)),window_correlation_median=float(np.median(corr)),edf_window_rms_median=float(np.median(rmsx)),mat_window_rms_median=float(np.median(rmsy)),max_absolute_sample_difference=float(abs(y-x).max()),equal_samples=bool(np.array_equal(x,y)));rows.append(rec)
  for t in range(n):windows.append(dict(channel=name,epoch=t,edf_rms=rmsx[t],mat_rms=rmsy[t],edf_std=np.sqrt(vx[t]),mat_std=np.sqrt(vy[t]),correlation=float(cross[t]/np.sqrt(vx[t]*vy[t])) if valid[t] else None))
  freq,px=welch(x,fs=200,nperseg=1024,noverlap=512,axis=-1);_,py=welch(y,fs=200,nperseg=1024,noverlap=512,axis=-1);px=px.mean(0);py=py.mean(0);psds[name+"_edf"]=px;psds[name+"_mat"]=py
  for lo,hi in [(0,.5),(.5,4),(4,8),(8,15),(15,35),(35,90)]:
   mask=(freq>=lo)&(freq<hi);bands.append(dict(channel=name,low_hz=lo,high_hz=hi,mat_over_edf_power=float(py[mask].sum()/px[mask].sum())))
  for t in [0,100,400,800]:snippets[f"{name}_edf_epoch{t}"]=x[t];snippets[f"{name}_mat_epoch{t}"]=y[t]
  print("CHANNEL VERIFIED",rec,flush=True)
pd.DataFrame(rows).to_csv(O/"channel_comparison.csv",index=False);pd.DataFrame(windows).to_csv(O/"window_comparison.csv",index=False);pd.DataFrame(bands).to_csv(O/"frequency_band_ratios.csv",index=False);np.savez_compressed(O/"comparison_snippets.npz",**snippets);np.savez_compressed(O/"mean_psd.npz",frequency_hz=freq,**psds)
audit=dict(status="PASS",subject="sub-III001",official_mat_meta_mac_verified=True,official_mat_sha256=download["plaintext_sha256"],existing_edf_sha256=sha(edf),channels=6,overlap_epochs=n,full_edf_epochs=total//6000,official_last30_omission_confirmed=True,mat_unit_not_independently_declared=True,signals_not_sample_identical=True,numerical_comparison_only=True,no_units_changed=True,no_models_or_predictions_changed=True,scope="One prespecified subject only; does not establish acquisition calibration or identity for remaining9healthy subjects.",retained_snippets_sha256=sha(O/"comparison_snippets.npz"),source_files_still_present_at_audit=True)
(O/"comparison_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>官方MAT与既有EDF：首位健康受试者跨格式核对</h1><p>预定subject1，不按波形或性能选人。官方公开MAT经MEGA原始分块MAC校验，既有EDF再次核对清单SHA256。六个参考EEG通道按同起点、200Hz、30秒窗口比较全部重叠部分。MAT有924窗，EDF有954窗，恰少末尾30窗，与<a href="https://sleeptight.isr.uc.pt/?page_id=76">官方说明</a>一致。没有读取MAT睡眠标签或用其重建评分时间轴。</p>'+pd.DataFrame(rows).to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>频段功率比（MAT/EDF）</h2>'+pd.DataFrame(bands).to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>MAT没有独立的单位声明变量，因此这里只能比较数值尺度与波形，不能据此证明设备真实物理标定。两格式并非逐样本一致，频谱与幅值差异可能来自提取预处理；未经官方明确说明，不能认定具体滤波器。单人结果不能推广到全部健康组。既有EDF单位、导联、完整评分窗口和所有模型预测均未更改。</p><p><a href="download_audit.json">官方文件MAC/SHA256</a> · <a href="comparison_audit.json">比较审计</a> · <a href="channel_comparison.csv">各通道统计</a> · <a href="window_comparison.csv">逐窗统计</a> · <a href="comparison_snippets.npz">固定四窗可复核片段</a></p>')
print(audit,flush=True)

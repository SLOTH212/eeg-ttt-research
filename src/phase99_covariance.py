from pathlib import Path
import json,hashlib
import numpy as np
from scipy.signal import lfilter
R=Path(__file__).resolve().parents[1];O=R/"phase99"
def moments(raw):
 x=np.asarray(raw,dtype=np.float64);return np.einsum("nct,ndt->ncd",x,x,optimize=False)/x.shape[-1]
def root(a,power):
 v,u=np.linalg.eigh(a);assert v.min()>0;return (u*(v**power))@u.T
def transform(raw,gains,reference,mode):
 assert mode in ["diagonal","full"];raw=np.asarray(raw);gains=np.asarray(gains);assert raw.ndim==3 and raw.shape[1]==2 and len(raw)==len(gains)
 cov=moments(raw);tr=np.trace(cov,axis1=1,axis2=2);valid=tr>=2e-12;shape=np.zeros_like(cov);shape[valid]=2*cov[valid]/tr[valid,None,None];state=None;states=np.full_like(cov,np.nan);out=raw.copy();correction=np.ones(len(raw));active=(gains>1)&valid;source=.99*reference+.01*np.eye(2);sr=root(source,.5);maxerr=0.;clipped=0
 for t in range(len(raw)):
  if valid[t]:state=shape[t].copy() if state is None else .99*state+.01*shape[t]
  if state is not None:states[t]=state
  if not active[t]:continue
  target=.99*state+.01*np.eye(2)
  matrix=sr@root(target,-.5) if mode=="full" else np.diag(np.sqrt(np.diag(source)/np.diag(target)))
  x=raw[t].astype(float);z=matrix@x;oldenergy=float(np.mean(x*x));newenergy=float(np.mean(z*z));assert newenergy>0;scale=np.sqrt(oldenergy/newenergy);correction[t]=scale;z*=scale*gains[t];relative_error=abs(float(np.mean(z*z))-oldenergy*gains[t]**2)/max(oldenergy*gains[t]**2,1e-30);maxerr=max(maxerr,relative_error);clipped+=int((abs(z)>20).sum());out[t]=np.clip(z,-20,20).astype(np.float32)
 return out,dict(shape_states=states,energy_corrections=correction,active=active,changed=np.any(out!=raw,axis=(1,2)),gains=gains),dict(max_relative_energy_error_beforeclip=maxerr,clipped_samples=clipped)
def setup():
 O.mkdir(exist_ok=True);old=json.loads((R/"phase93/protocol.json").read_text());subjects=old["train"]
 pr=dict(purpose="Label-free causal channel-shape controls added to alreadyfixed scalar low-amplitude guard. Source-only reference, no gradients or classifierupdates.",source_subjects=subjects,source_reference="Perwindow two-channel secondmoment normalizedto trace2; meanwithinperson then equalmeanover10sourcepeople, no labels.",target_state="EMA.01 of trace2 current-window secondmoments; initializefirst nonflatwindow, skipflatRMS<1e-6. Currentandpast only.",regularization="Bothsourceandtarget shape matrices shrink .01towardidentity.",modes=["diagonal","full"],transforms="Diagonal sqrt(diagSsource/diagStarget), or Ssource^0.5 Starget^-0.5. Rescale each transformed currentwindow to retain exact scalar-guard jointRMS before clipping.",activation="Only windows where existingphase79guard gain>1; allotherinputsexactunchanged. Identicalexistinggains, boundsandclipping.",selection="Fixed design,no matrix/EMA/shrinkage parametersearch; all targetpeoplepreviouslyobserved. Startwith algebraandcausalityaudit.",scalar_protocol_sha256=hashlib.sha256((R/"phase79/protocol.json").read_bytes()).hexdigest(),extra_class_labels=0,interpretation="Custom causal/source-colored/RMS-preserving variant inspiredbyEA; not exact originalalgorithm. Sourcewindowpoolmayincludeunscoredwindows.",reference_url="https://arxiv.org/pdf/1808.05464v2",reference_scope="Original EA uses subject mean covariance inverse square root; studied motorimagery/ERP, not our sleepstaging setting.")
 (O/"protocol.json").write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest());individual=[];records=[]
 for sid in subjects:
  folder=R/("data/processed" if sid<20 else "phase2/processed");path=folder/f"s{sid:02d}_x.npy";x=np.load(path,mmap_mode="r");cc=[]
  for i in range(0,len(x),128):cc.append(moments(x[i:i+128]))
  cov=np.concatenate(cc);tr=np.trace(cov,axis1=1,axis2=2);valid=tr>=2e-12;shape=2*cov[valid]/tr[valid,None,None];individual.append(shape.mean(0));records.append(dict(subject=sid,windows=len(x),nonflat=int(valid.sum()),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));print("SOURCE SHAPE",sid,len(x),flush=True)
 reference=np.mean(individual,axis=0);assert abs(np.trace(reference)-2)<1e-12;np.savez(O/"source_reference.npz",reference=reference,individual=np.array(individual),subjects=np.array(subjects));(O/"source_records.json").write_text(json.dumps(records,indent=2))
 # Independent 2x2 closed-form roots and covariance identities.
 rng=np.random.default_rng(99017);errors=[]
 def closed(a):d=np.sqrt(np.linalg.det(a));return (a+d*np.eye(2))/np.sqrt(np.trace(a)+2*d)
 for _ in range(50):
  u=rng.normal(size=(2,2));target=u@u.T+.01*np.eye(2);source=.99*reference+.01*np.eye(2);matrix=root(source,.5)@root(target,-.5);independent=closed(source)@np.linalg.inv(closed(target));err=float(abs(matrix-independent).max());alignerr=float(abs(matrix@target@matrix.T-source).max());assert err<1e-10 and alignerr<1e-10;errors.append([err,alignerr])
 checks=[]
 for sid,rawpath,gainpath in [("source_s05",R/"data/processed/s05_x.npy",R/"phase79/predictions/source_s05_native_guarded_median_normalizer.npz"),("healthy_III001",R/"phase76/processed/sub-III001_left_x.npy",R/"phase79/predictions/external_sub-III001_left_guarded_median_normalizer.npz")]:
  raw=np.load(rawpath,mmap_mode="r");gains=np.load(gainpath)["gains"];cov=moments(raw);tr=np.trace(cov,axis1=1,axis2=2);valid=tr>=2e-12;sh=2*cov[valid]/tr[valid,None,None];filtered,_=lfilter([.01],[1,-.99],sh,axis=0,zi=.99*sh[:1])
  for mode in pr["modes"]:
   x,aux,stats=transform(raw,gains,reference,mode);prefix=transform(raw[:65],gains[:65],reference,mode)[0];future=raw[:66].copy();future[65]*=13;future[65,0]+=3;perturbed=transform(future,gains[:66],reference,mode)[0];assert np.array_equal(prefix,x[:65]) and np.array_equal(perturbed[:65],x[:65]);err=float(abs(aux["shape_states"][valid]-filtered).max());assert err<1e-12 and stats["max_relative_energy_error_beforeclip"]<1e-12 and np.array_equal(x[~aux["active"]],raw[~aux["active"]]);checks.append(dict(case=sid,mode=mode,prefix_exact=True,future_perturbation_exact=True,scipy_state_error=err,unchanged_input_exact=True,**stats))
 zero=np.zeros((4,2,3000),np.float32);z,za,_=transform(zero,np.ones(4),reference,"full");assert np.array_equal(z,zero)
 singular=np.repeat(rng.normal(size=(4,1,3000)).astype(np.float32),2,axis=1);z,za,zs=transform(singular,np.full(4,2.),reference,"full");assert np.isfinite(z).all() and zs["max_relative_energy_error_beforeclip"]<1e-12
 audit=dict(status="PASS",source_people=10,source_windows=sum(v["windows"] for v in records),labels_notread=True,closedform_root_max_error=float(np.array(errors)[:,0].max()),covariance_mapping_max_error=float(np.array(errors)[:,1].max()),real_causal_checks=checks,flat_and_singular_cases_pass=True,source_reference=reference.tolist());(O/"audit.json").write_text(json.dumps(audit,indent=2));print(audit)
 (O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:35px auto}</style><h1>固定幅值触发下的因果通道形状校正：实现审计</h1><p>只用10位源训练者的无标签信号构造二维参考。沿用既有低幅值触发和增益，额外校正两个通道的相对强弱或完整二阶关系，并保持校正后当前窗口的联合RMS与纯标量方法相同；未触发窗口逐元素不变。</p><p>本设计借鉴<a href="https://arxiv.org/pdf/1808.05464v2">He与Wu的Euclidean Alignment</a>中按参考协方差逆平方根变换信号的思路。原文研究运动想象与ERP；本实验是面向现有睡眠模型的因果变体，采用源参考着色、固定触发和每窗能量保持。这里仅证明实现与因果性，尚未证明分类收益。</p><pre>'+json.dumps(audit,indent=2)+'</pre><p><a href="protocol.json">固定方法</a> · <a href="audit.json">独立代数与前缀审计</a></p>')
if __name__=="__main__":setup()

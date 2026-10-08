from common import *
from phase2_temporal import ema
from scipy.signal import lfilter
import pandas as pd,hashlib
O=ROOT/"phase85";pr=json.loads((O/"protocol.json").read_text());assert (O/"complete.json").exists();assert json.loads((O/"prediction_audit.json").read_text())["status"]=="PASS"
assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text()
base=json.loads((ROOT/"phase79/protocol.json").read_text());assert hashlib.sha256((ROOT/"phase79/protocol.json").read_bytes()).hexdigest()==pr["normalizer_protocol_sha256"]
for m in pr["models"]:assert hashlib.sha256(Path(m["path"]).read_bytes()).hexdigest()==m["sha256"]
rows=[];norm=[];maxerr=0.
for c in pr["cases"]:
 name="_".join(c[k] for k in ["domain","subject","montage"]);yp=np.load(c["labels"]);y=yp["y"] if c["labels"].endswith(".npz") else yp;raw=np.load(c["x"],mmap_mode="r");assert len(raw)==len(y);rms=np.sqrt(np.mean(raw.astype(float)**2,axis=(1,2)))
 for mode in pr["modes"]:
  d=np.load(O/"predictions"/f"{name}_{mode}.npz");assert set(d.files)=={f"eegnet_seed{s}" for s in [17,29,43]}
  if mode!="raw":
   a=np.load(O/"predictions"/f"{name}_{mode}_normalizer.npz");valid=rms>=1e-6;logs=np.log(rms[valid]);state,_=lfilter([.01],[1,-.99],logs,zi=[.99*logs[0]]);est=np.exp(state)
   gg=np.where(est<base["lower_source_rms_quantile"],base["source_reference_rms"]/est,1.) if mode=="guarded_median" else np.maximum(1,base["lower_source_rms_quantile"]/est)
   g=np.ones(len(raw));g[valid]=np.clip(gg,1,100);err=float(abs(g-a["gains"]).max());maxerr=max(maxerr,err);assert err<1e-10 and np.array_equal(rms,a["current_rms"]);norm.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,modified_windows=int((g!=1).sum()),windows=len(g)))
  for key in d.files:
   p=d[key];assert p.shape==(len(y),5)
   for alpha in pr["ema"]:
    q=p if alpha==1 else ema(p,alpha);m=metrics(y,q);cm=np.array(m["confusion"]);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,model="eegnet",seed=int(key.split("seed")[-1]),ema=alpha,macro_f1=100*m["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=m["ce"]))
df=pd.DataFrame(rows);assert len(df)==2484;df.to_csv(O/"participants.csv",index=False);pd.DataFrame(norm).to_csv(O/"normalizer_people.csv",index=False)
person=df.groupby(["domain","subject","montage","mode","model","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);su=person.groupby(["domain","montage","mode","model","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();su.to_csv(O/"summary.csv",index=False)
refs=[];a=pd.read_csv(ROOT/"phase79/seedmean_people.csv");a["domain"]=a.domain.map({"source":"source","external":"healthy"});refs.append(a)
a=pd.read_csv(ROOT/"phase81/seedmean_people.csv");a["domain"]="sleepedf";a["montage"]="native";a["subject"]=a.subject.map(lambda v:f"s{int(v):02d}");refs.append(a)
a=pd.read_csv(ROOT/"phase83/seedmean_people.csv");a["domain"]="patient";refs.append(a)
ref=pd.concat(refs,ignore_index=True);rng=np.random.default_rng(85017);comparisons=[]
for (domain,montage,mode,alpha),g in person.groupby(["domain","montage","mode","ema"]):
 for other in ["cbramod","labram","context11"]:
  b=ref[(ref.domain==domain)&(ref.montage==montage)&(ref["mode"]==mode)&(ref.ema==alpha)&(ref.model==other)]
  if b.empty:continue
  pair=g.merge(b,on="subject",suffixes=("_eegnet","_other"),validate="one_to_one");assert len(pair)==len(g);v=(pair.macro_f1_eegnet-pair.macro_f1_other).to_numpy();ci=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975]);comparisons.append(dict(domain=domain,montage=montage,mode=mode,ema=alpha,comparator=other,people=len(v),eegnet_f1=pair.macro_f1_eegnet.mean(),other_f1=pair.macro_f1_other.mean(),difference_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],positive=int((v>0).sum()),harm_over_1pp=int((v< -1).sum()),worst_pp=v.min()))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"paired_comparisons.csv",index=False)
audit=dict(status="PASS",cases=92,method_streams_recomputed=2484,normalizer_cases=len(norm),max_gain_error=maxerr,source_checkpoints_unchanged=True,all_target_labels_previously_observed=True,not_blind_validation=True,no_training_or_hyperparameter_search=True,no_ttt=True)
(O/"audit.json").write_text(json.dumps(audit,indent=2));tab=su[su.ema==.5];comparison=comp[(comp.ema==.5)&comp.montage.isin(["native","left"])]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>历史EEGNet冻结模型：跨队列外部对照</h1><p>复用同一10位混合源训练者、4位源验证者选出的三个历史检查点。该模型为EEGNet-style 8,2、100Hz输入、adaptive average pool8，非官方逐项复现。原训练预算25轮，与基础模型10轮不完全匹配；本轮无训练、梯度更新或目标调参。所有目标标签此前已观察，本轮仅补充架构对照，不能声称新的盲测。</p><p>raw与固定guard/floor输入处理，三种固定平滑强度全部报告；同方法、同EMA逐人配对，并先在人内平均三个模型种子。外部导联为侧方近似映射，仍不等同训练导联。</p>'+tab.to_html(index=False,float_format=lambda v:f"{v:.4f}")+'<h2>同EMA0.5及同输入处理的架构差异：EEGNet减对照</h2>'+comparison.to_html(index=False,float_format=lambda v:f"{v:.4f}")+'<p><a href="audit.json">指标与输入复算</a> · <a href="prediction_audit.json">CPU/GPU一致性及冻结状态</a> · <a href="protocol.json">方案与历史模型哈希</a> · <a href="paired_comparisons.csv">含右导联及全部平滑设置</a></p>')
print(tab.to_string(index=False));print(comparison.to_string(index=False));print(audit)

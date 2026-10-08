from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase100";pr=json.loads((O/"protocol.json").read_text());assert (O/"complete.json").exists();assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text()
rows=[];stages=[]
for c in pr["cases"]:
 name="_".join(c[k] for k in ["domain","subject","montage"]);a=json.loads((O/"predictions"/(name+"_audit.json")).read_text());lp=Path(a["baseline_path"]);assert hashlib.sha256(lp.read_bytes()).hexdigest()==a["baseline_sha256"]
 yy=np.load(c["labels"]);y=yy["y"] if c["labels"].endswith(".npz") else yy
 for mode in ["scalar"]+pr["modes"]:
  z=np.load(lp if mode=="scalar" else O/"predictions"/f"{name}_{mode}.npz")
  for spec in pr["models"]:
   kind=spec["model"];seed=spec["seed"];p=z[f"{kind}_seed{seed}"]
   for alpha in [1.,.8,.5]:
    q=p if alpha==1 else ema(p,alpha);mm=metrics(y,q);cm=np.array(mm["confusion"]);base=dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,model=kind,seed=seed,ema=alpha)
    rows.append(dict(**base,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],ce=mm["ce"],accuracy=100*cm.trace()/cm.sum()))
    if alpha==.5:
     for k,stage in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(**base,stage=stage,f1=200*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
df=pd.DataFrame(rows);assert len(df)==7452;df.to_csv(O/"participants.csv",index=False);keys=["domain","subject","montage","mode","model","ema"];assert (df.groupby(keys).seed.nunique()==3).all()
person=df.groupby(keys)[["macro_f1","balanced_accuracy","ce","accuracy"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False)
summary=person.groupby(["domain","montage","mode","model","ema"]).macro_f1.mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comp=[];rng=np.random.default_rng(100017)
for (domain,montage,model,alpha),g in person.groupby(["domain","montage","model","ema"]):
 p=g.pivot(index="subject",columns="mode",values="macro_f1");idx=rng.integers(0,len(p),(10000,len(p)))
 for mode in ["diagonal","full"]:
  v=p[mode]-p.scalar;ci=np.quantile(v.to_numpy()[idx].mean(1),[.025,.975]);comp.append(dict(domain=domain,montage=montage,model=model,ema=alpha,mode=mode,people=len(v),new_f1=p[mode].mean(),scalar_f1=p.scalar.mean(),difference_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((v>0).sum()),harm_over_1pp=int((v< -1).sum()),worst_pp=v.min()))
comp=pd.DataFrame(comp);comp.to_csv(O/"paired_comparisons.csv",index=False);primary=comp[(comp.model=="cbramod")&(comp["mode"]=="full")&(comp.ema==.5)&comp.montage.isin(["native","left"])];primary.to_csv(O/"primary_results.csv",index=False)
pd.DataFrame(stages).to_csv(O/"stage_participants.csv",index=False)
audit=dict(status="PASS",streams=len(df),case_count=92,three_seeds_averaged_within_person=True,no_probability_ensemble=True,no_new_blind_claim=True,no_gradient_updates=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>相同标量幅值校正下的通道形状对照</h1><p>冻结CBraMod、LaBraM及context11各三个源检查点；使用同一旧幅值触发与增益。新方法额外校正两路相对强度（diagonal）或二阶关系（full），每窗重标定以保持裁剪前联合RMS相同。仅使用当前及历史无标签窗口，未触发输入与受影响历史范围之外的预测保持不变。</p><p>以下固定主比较为CBraMod full相对既有scalar，EMA0.5，原生SleepEDF或左侧ISRUC。所有目标此前已经观察，属于探索性表征；未针对结果选择参数。三个种子在人内平均指标，区间按人配对重采样，不合并人群。</p>'+primary.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>EMA0.5全部对照</h2>'+comp[comp.ema==.5].to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p>二维形状改变可能影响空间信息，联合RMS相同不等于全部统计相同；裁剪后的能量差异另行记录。本实验属于无梯度输入对照，不代表测试时训练成功。</p><p><a href="audit.json">预测与因果审计</a> · <a href="summary_audit.json">统计审计</a> · <a href="../phase99/report.html">源参考及代数审计</a></p>')
print(primary.to_string(index=False));print(audit)

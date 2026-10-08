from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/"phase84";O.mkdir(exist_ok=True)
files=[R/f"phase{p}/seedmean_people.csv" for p in [79,81,83]]
frames=[]
a=pd.read_csv(files[0])
for dom,label in [("source","Source validation"),("external","ISRUC healthy")]:
 d=a[(a.domain==dom)&(a.montage==("native" if dom=="source" else "left"))].copy();d["cohort"]=label;frames.append(d)
d=pd.read_csv(files[1]);d["subject"]=d.subject.map(lambda x:f"s{int(x):02d}");d["cohort"]="SleepEDF observed";frames.append(d)
d=pd.read_csv(files[2]);d=d[d.montage=="left"].copy();d["cohort"]="ISRUC patient";frames.append(d)
allp=pd.concat(frames,ignore_index=True);allp=allp[(allp.ema==.5)&allp.model.isin(["cbramod","labram","context11"])&allp["mode"].isin(["raw","guarded_median","floor"])]
normal={}
for dom,label,sids in [("source","Source validation",[f"s{i:02d}" for i in [5,15,25,35]]),("external","ISRUC healthy",[f"sub-III{i:03d}" for i in range(1,11)])]:
 for sid in sids:
  for mode in ["guarded_median","floor"]:
   f=R/"phase79/predictions"/f"{dom}_{sid}_{'native' if dom=='source' else 'left'}_{mode}_normalizer.npz";n=np.load(f);g=n["gains"];normal[(label,sid,mode)]=(float((g!=1).mean()),float(g.max()),float(np.median(n["current_rms"]))*100)
for r in pd.read_csv(R/"phase81/normalizer_people.csv").itertuples():
 sid=f"s{r.subject:02d}";f=R/"phase81/predictions"/f"{sid}_{r.mode}_normalizer.npz";n=np.load(f);normal[("SleepEDF observed",sid,r.mode)]=(r.changed_fraction,r.gain_max,float(np.median(n["current_rms"]))*100)
for r in pd.read_csv(R/"phase83/normalizer_people.csv").itertuples():
 if r.montage!="left":continue
 n=np.load(R/"phase83/predictions"/f"{r.subject}_left_{r.mode}_normalizer.npz")
 normal[("ISRUC patient",r.subject,r.mode)]=(r.modified_windows/r.windows,r.gain_max,float(np.median(n["current_rms"]))*100)
rows=[]
for (cohort,model),g in allp.groupby(["cohort","model"]):
 p=g.pivot(index="subject",columns="mode",values="macro_f1")
 for sid,r in p.iterrows():
  for mode in ["guarded_median","floor"]:
   frac,gmax,rms=normal[(cohort,sid,mode)]
   rows.append(dict(cohort=cohort,model=model,subject=sid,mode=mode,raw_f1=r.raw,adjusted_f1=r[mode],delta_pp=r[mode]-r.raw,modified_fraction=frac,max_gain=gmax,median_joint_rms_uv=rms))
df=pd.DataFrame(rows);df.to_csv(O/"person_risk.csv",index=False);rng=np.random.default_rng(84017);summary=[]
for (cohort,model,mode),g in df.groupby(["cohort","model","mode"]):
 v=g.delta_pp.to_numpy();ci=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975]);active=g.modified_fraction>0
 assert (abs(g.loc[~active,"delta_pp"])<1e-10).all()
 summary.append(dict(cohort=cohort,model=model,mode=mode,n=len(g),modified_people=int(active.sum()),positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),harm_over_1pp=int((v< -1).sum()),raw_f1=g.raw_f1.mean(),adjusted_f1=g.adjusted_f1.mean(),delta_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],worst_pp=v.min(),best_pp=v.max(),modified_people_mean_pp=g.loc[active,"delta_pp"].mean()))
s=pd.DataFrame(summary);s.to_csv(O/"summary.csv",index=False)
order=["Source validation","SleepEDF observed","ISRUC healthy","ISRUC patient"];models=["cbramod","labram","context11"];colors=["#277DA1","#F8961E"]
fig,axes=plt.subplots(1,3,figsize=(15,5.5),sharey=True)
for ax,model in zip(axes,models):
 for j,mode in enumerate(["guarded_median","floor"]):
  t=s[(s.model==model)&(s["mode"]==mode)].set_index("cohort").loc[order];yy=np.arange(4)+(j-.5)*.22
  ax.errorbar(t.delta_pp,yy,xerr=np.array([t.delta_pp-t.ci_low,t.ci_high-t.delta_pp]),fmt="o",capsize=4,label=mode,color=colors[j])
 ax.axvline(0,color="#555",lw=.8);ax.set_title(model);ax.set_xlabel("Macro-F1 change (percentage points)");ax.grid(axis="x",alpha=.2);ax.set_yticks(range(4),order);ax.invert_yaxis()
axes[0].legend(loc="best",fontsize=9);fig.suptitle("Fixed amplitude controls: paired participant means, 3 seeds averaged within person\nSame EMA=0.5; ISRUC left proxy montage; conditional 95% participant bootstrap",fontsize=12);fig.tight_layout();fig.savefig(O/"cross_cohort_risk.png",dpi=170);plt.close(fig)
patient=df[(df.cohort=="ISRUC patient")&(df.model=="cbramod")&(df["mode"]=="guarded_median")].sort_values("delta_pp")
inputs={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};audit=dict(status="PASS",source_files_sha256=inputs,cohort_sizes={c:int(df[(df.cohort==c)&(df.model=="cbramod")&(df["mode"]=="guarded_median")].shape[0]) for c in order},no_new_tuning=True,all_analyses_retrospective=True,unchanged_inputs_have_zero_f1_change=True,no_cohort_pooling=True,no_new_hypothesis_confirmation=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}img{width:100%}</style><h1>固定幅值处理：跨人群收益与伤害</h1><p>不改变任何阈值、模型、导联或方法。原始结果按同一EMA0.5、三个种子人内平均的口径汇总；外部队列均展示预定左侧近似导联。本报告是结果已知后的描述性风险分析，不把多个队列合并成一次成功的确认性检验。19位患者的预定主比较区间跨0，仍保留为未获明确支持。</p><img src="cross_cohort_risk.png"><h2>逐队列比较</h2>'+s.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<h2>新患者主方法逐人结果</h2>'+patient.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<p>低幅值触发比例是输入特征，不是安全保证。触发者内均值属于事后子组描述；不得据此选择人群或反向调整阈值。健康队列曾用于方法开发；SleepEDF目标者此前也已观察。区间仅反映固定模型种子和现有受试者样本的不确定性。</p><p><a href="../phase83/report.html">独立患者原始结果</a> · <a href="audit.json">汇总审计</a> · <a href="person_risk.csv">逐人完整表</a></p>')
print(s[s.model=="cbramod"].to_string(index=False));print(patient.to_string(index=False));print(audit)

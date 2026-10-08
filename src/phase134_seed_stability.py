
import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/"phase134";O.mkdir(exist_ok=True)
inputs=[R/"phase131/new_streams.csv",R/"phase120/participants.csv",R/"phase123/participants.csv"]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(purpose="Descriptive seed and leave-one-person-out stability of frozen history-versus-repeat effects, original and alternate head selectors",inputs={str(p):sha(p) for p in inputs},conditions="EMA .5; SleepEDF native/raw; healthy and oldpatient left/full",scope="No new inference, training, chosen seed, selector or target tuning. Three seeds are paired fits on same people, not independent participant sample sizes. Leave-one-person-out range is a sensitivity range, not a confidence interval.")
(O/"protocol.json").write_text(json.dumps(protocol,indent=2))
frames=[pd.read_csv(inputs[0])]
for back,path in zip(["cbramod","labram"],inputs[1:]):
 d=pd.read_csv(path);d["backbone"]=back;d["selector"]="original";frames.append(d)
d=pd.concat(frames,ignore_index=True)
d=d[(d.ema==.5)&d.variant.isin(["history11","repeat11"])&(((d.domain=="sleepedf")&(d.montage=="native")&(d["mode"]=="raw"))|((d.domain.isin(["healthy","patient"]))&(d.montage=="left")&(d["mode"]=="full")))]
index=["backbone","domain","selector","subject","seed"]
w=d.pivot(index=index,columns="variant",values="macro_f1").reset_index()
assert len(w)==2*2*3*59 and w[["history11","repeat11"]].notna().all().all()
w["effect_pp"]=w.history11-w.repeat11;w.to_csv(O/"paired_seed_people.csv",index=False)
seedrows=[];personrows=[];robust=[]
for keys,g in w.groupby(["backbone","domain","selector"]):
 back,domain,selector=keys
 for seed,s in g.groupby("seed"):
  z=s.effect_pp.to_numpy()
  seedrows.append(dict(backbone=back,domain=domain,selector=selector,seed=seed,people=len(z),effect_pp=z.mean(),positive_people=int((z>0).sum()),harm_over1pp=int((z< -1).sum()),worst_pp=z.min()))
 p=g.groupby("subject").effect_pp.agg(["mean","min","max"]).reset_index()
 for row in p.to_dict("records"):
  personrows.append(dict(backbone=back,domain=domain,selector=selector,subject=row["subject"],mean_effect_pp=row["mean"],worst_seed_effect_pp=row["min"],best_seed_effect_pp=row["max"],all_seeds_positive=bool(row["min"]>0),all_seeds_negative=bool(row["max"]<0)))
 z=p["mean"].to_numpy();loo=(z.sum()-z)/(len(z)-1)
 sm=g.groupby("seed").effect_pp.mean()
 robust.append(dict(backbone=back,domain=domain,selector=selector,people=len(z),effect_pp=z.mean(),median_person_effect_pp=np.median(z),min_seed_mean_pp=sm.min(),max_seed_mean_pp=sm.max(),positive_seed_means=int((sm>0).sum()),all_seeds_positive_people=int((p["min"]>0).sum()),all_seeds_negative_people=int((p["max"]<0).sum()),loo_mean_min_pp=loo.min(),loo_mean_max_pp=loo.max(),all_loo_means_positive=bool((loo>0).all())))
sf=pd.DataFrame(seedrows);rf=pd.DataFrame(robust)
sf.to_csv(O/"seed_summary.csv",index=False);rf.to_csv(O/"robustness.csv",index=False);pd.DataFrame(personrows).to_csv(O/"people_agreement.csv",index=False)
fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True)
colors={17:"#1677b8",29:"#e18922",43:"#239b6f"}
for i,back in enumerate(["cbramod","labram"]):
 for j,domain in enumerate(["sleepedf","healthy","patient"]):
  ax=axes[i,j];q=sf[(sf.backbone==back)&(sf.domain==domain)]
  for k,seed in enumerate([17,29,43]):
   yy=[q[(q.selector==sel)&(q.seed==seed)].effect_pp.item() for sel in ["original","alternate"]]
   ax.plot([0,1],yy,"o-",color=colors[seed],alpha=.85,label=f"Seed {seed}")
  means=[rf[(rf.backbone==back)&(rf.domain==domain)&(rf.selector==sel)].effect_pp.item() for sel in ["original","alternate"]]
  ax.plot([0,1],means,"D--",color="black",markersize=5,label="Mean of seeds")
  ax.axhline(0,color="gray",linewidth=.8);ax.set_xticks([0,1],["Original","Alternate"]);ax.set_xlim(-.2,1.2);ax.grid(axis="y",alpha=.2)
  ax.set_title(f"{'CBraMod' if i==0 else 'LaBraM'} / {domain}")
  
  if i==1:ax.set_xlabel("Head checkpoint selector")
fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,fontsize=8)
fig.supylabel("History minus repeat Macro-F1 (pp)",fontsize=10)
fig.suptitle("Static history benefit: sensitivity to seed and source head selector",fontsize=13)
fig.text(.5,.01,"Same observed people; fixed parent checkpoints; EMA 0.5. External cohorts use left montage + full input calibration.",ha="center",fontsize=9)
fig.tight_layout(rect=[.025,.035,1,.87]);fig.savefig(O/"seed_selector_stability.png",dpi=145);plt.close(fig)
audit=dict(status="PASS",paired_person_seed_rows=len(w),seed_groups=len(sf),person_groups=len(personrows),no_new_inference=True,input_hashes_rechecked=all(sha(p)==protocol["inputs"][str(p)] for p in inputs))
assert audit["input_hashes_rechecked"]
(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%}</style><h1>静态历史效应的种子与单人敏感性</h1><p>所有原/替代源头选模组、两个基础模型及三个种子均保留。每个种子先在同一批受试者内计算历史头减等容量重复当前窗口头的 Macro-F1 差；种子均值不是额外独立受试者。只描述既有观察结果，没有重训、挑种子或换主检验。</p><img src="seed_selector_stability.png"><p>逐人先平均三个种子，再计算中位数与每次剔除一人的均值范围；此范围不是置信区间。原始全人群配对区间见 Phase131。父模型仍使用原始源组选择，并非完整流程换折。</p>'+rf.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<h2>各个种子</h2>'+sf.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<p><a href="../phase131/report.html">主敏感性评价</a></p>')
print(rf.to_string(index=False));print(audit)

from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.signal import lfilter
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];pr=json.loads((r/"phase79/protocol.json").read_text());logs=[];weights=[]
for s in pr["source_reference_subjects"]:
 x=np.load(r/("data/processed" if s<20 else "phase2/processed")/f"s{s:02d}_x.npy",mmap_mode="r");v=np.log(np.maximum(np.sqrt(np.mean(x.astype(float)**2,axis=(1,2))),1e-6));z,_=lfilter([.01],[1,-.99],v,zi=[.99*v[0]]);logs.extend(z);weights.extend([1/len(z)]*len(z))
q=np.load(r/"phase79/source_log_rms_distribution.npz");err=float(np.max(abs(np.array(logs)-q["values"])));assert err<1e-12;assert np.array_equal(weights,q["weights"]);order=np.argsort(logs);w=np.array(weights)[order];bound=float(np.exp(np.array(logs)[order][np.searchsorted(np.cumsum(w)/w.sum(),.01)]));assert abs(bound-pr["lower_source_rms_quantile"])<1e-12
(r/"phase79/source_threshold_audit.json").write_text(json.dumps(dict(status="PASS",independent_scipy_logstate_error=err,source_weighting_exact=True,recomputed_lower_rms=bound),indent=2))
a=pd.read_csv(r/"phase78/seedmean_people.csv");b=pd.read_csv(r/"phase79/seedmean_people.csv");a=a[a["mode"].isin(["raw","logema01"])];b=b[b["mode"].isin(["guarded_median","floor"])];d=pd.concat([a,b]);d=d[d.ema==.5];fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True);colors={"cbramod":"#1b6aa5","labram":"#db853b","context11":"#30927a"};modes=["raw","logema01","guarded_median","floor"];labels=["Raw","Always\nRMS match","Guarded\nRMS match","Amplitude\nfloor"]
for ax,(domain,montage,title) in zip(axes,[("source","native","SleepEDF source validation (4 people)"),("external","left","ISRUC left proxy montage (10 people)"),("external","right","ISRUC right proxy montage (10 people)")]):
 sub=d[(d.domain==domain)&(d.montage==montage)]
 for j,(model,color) in enumerate(colors.items()):
  means=[sub[(sub.model==model)&(sub["mode"]==mode)].macro_f1.mean() for mode in modes];pos=np.arange(4)+(j-1)*.24;ax.bar(pos,means,width=.22,color=color,label=model,alpha=.85)
  for xx,mode,mean in zip(pos,modes,means):
   yy=sub[(sub.model==model)&(sub["mode"]==mode)].macro_f1.to_numpy();ax.scatter(xx+np.linspace(-.065,.065,len(yy)),yy,s=10,color=color,edgecolors="white",linewidths=.3,zorder=4)
 ref=sub[(sub.model=="relative")&(sub["mode"]=="raw")].macro_f1.mean();ax.axhline(ref,color="#4a4252",ls="--",lw=1.4,label="relative-power static");ax.set_xticks(range(4),labels);ax.set_title(title,fontsize=11);ax.set_ylim(0,100);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel("Per-person Macro-F1 (%)");handles,leg=axes[0].get_legend_handles_labels();fig.legend(handles,leg,loc="upper center",ncol=4,bbox_to_anchor=(.5,.96));fig.suptitle("Input amplitude controls: external gains do not imply safe source behavior",fontsize=14,y=1.02);fig.text(.5,-.005,"Bars: mean of person metrics after averaging 3 seeds. Dots: people. All models frozen; same output EMA=0.5. Posthoc diagnostic.",ha="center",fontsize=10);fig.tight_layout(rect=(0,.04,1,.90));fig.savefig(r/"phase79/amplitude_tradeoff.png",dpi=180,bbox_inches="tight");plt.close(fig)
p=r/"phase79/report.html";s=p.read_text();s+='<h2>源侧与外部的性能权衡</h2><img src="amplitude_tradeoff.png" style="max-width:100%" alt="Frozen-model amplitude controls across source and external data"><p><a href="source_threshold_audit.json">源幅值下界独立审计</a></p>';p.write_text(s)

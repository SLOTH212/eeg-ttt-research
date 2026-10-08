from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];out=r/"phase21";a=pd.read_csv(out/"pretraining_contrast.csv");b=pd.read_csv(out/"strong_control_comparisons.csv");b=b[(b.reference=="context11_ema_08")&(b.reference_seeds=="mean3seeds")&(b.method=="full_source_tuned_ema08")];fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True);labels=[]
for i,(source,cohort) in enumerate([(s,c) for s in ["young","mixed"] for c in ["young_dev","middle_dev","older"]]):
 labels.append(source+" source / "+cohort.replace("_dev", "")+" target");color="#2476a8" if source=="young" else "#d37b29"
 for ax,data,col in [(axes[0],a,"pretrained_minus_random_pp"),(axes[1],b,"delta_pp")]:
  v=data[(data.source==source)&(data.cohort==cohort)].iloc[0];ax.errorbar(v[col],i,xerr=[[v[col]-v.ci_low],[v.ci_high-v[col]]],fmt="o",color=color,capsize=4)
for ax,title in zip(axes,["Full training: pretrained vs random","Full pretrained + EMA vs history baseline + EMA"]):
 ax.axvline(0,color="#888",linestyle="--",linewidth=1);ax.set_title(title,fontsize=11);ax.set_xlabel("Paired Macro-F1 difference (percentage points)");ax.grid(axis="x",alpha=.2)
axes[0].set_yticks(range(6),labels);axes[0].invert_yaxis();fig.suptitle("Exploratory GPU experiments: participant-paired 95% bootstrap intervals",fontsize=12);fig.text(.5,.01,"One GPU training seed; history comparator averages three earlier seeds. Previously observed cohorts; no new blind test.",ha="center",fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(out/"comparison.png",dpi=160);plt.close(fig)

from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase60";df=pd.read_csv(o/"participants.csv")
fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
cases=[
("current",["frozen_ema0.5","current_continuous_blend1_ema0.5","current_decay0.999_blend1_ema0.5","static_current_ema0.5"],["Frozen","Continuous","Decay 0.999","Static (317 labels)"]),
("future",["frozen_ema0.5","future_continuous_blend1_ema0.5","future_decay0.99_blend1_ema0.5","static_future_ema0.5"],["Frozen","Continuous","Decay 0.99","Static (1,268 labels)"])]
names=[n for _,ns,_ in cases for n in ns];vals=df[df.method.isin(names)].macro_f1;lim=(np.floor(vals.min())-2,np.ceil(vals.max())+2)
colors=["#666666","#4477aa","#228833","#aa3377"]
for ax,(variant,names,labels) in zip(axes,cases):
 for i,(name,label) in enumerate(zip(names,labels)):
  vals=df[df.method==name].sort_values("subject").macro_f1.to_numpy()
  ax.hlines(vals.mean(),i-.32,i+.32,color=colors[i],linewidth=3,zorder=2)
  ax.scatter(i+np.linspace(-.14,.14,4),vals,color=colors[i],edgecolors="white",zorder=3,s=48)
  ax.text(i,vals.mean()+.35,f"{vals.mean():.2f}",ha="center",fontsize=10)
 ax.set_xticks(range(4),labels,rotation=16,ha="right");ax.set_title(variant.capitalize()+"-target auxiliary head")
 ax.set_ylim(*lim);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel("Macro-F1 (%)")
fig.suptitle("Simple memory controls versus label-matched static calibration",fontsize=14)
fig.text(.5,.01,"EMA 0.5 for all. Dots: 4 reused source-validation people; horizontal lines: means. Exploratory source-only results.",ha="center",fontsize=10)
fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(o/"memory_comparison.png",dpi=170);plt.close(fig)

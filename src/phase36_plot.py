from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];a=pd.read_csv(r/"phase35/participants.csv");d=pd.read_csv(r/"phase36/summary.csv");fig,axes=plt.subplots(2,3,figsize=(14,7));colors={"unfitted":"#a95b45","fitted":"#237f96"}
for row,source in enumerate(["young","mixed"]):
 g=a[a.source==source];ax=axes[row,0];vals=[g.initial_ce.mean(),g.position_prior_ce.mean(),g.shuffled_ce.mean(),g.fitted_ce.mean()];ax.bar(range(4),vals,color=["#adb6bc","#d0b576","#cf9184","#237f96"]);ax.set_xticks(range(4),["Unfitted","Position\nprior","Shuffled\ninput","Fitted"]);ax.set_ylabel(f"{source.capitalize()} source\nAuxiliary CE (lower is better)");ax.set_ylim(0,10)
 for j,v in enumerate(vals):ax.text(j,v+.1,f"{v:.2f}",ha="center",fontsize=9)
 for variant in ["unfitted","fitted"]:
  h=d[(d.source==source)&(d.variant==variant)&(d.objective=="masked_token")].sort_values("lr");axes[row,1].plot(h.lr,h.future_ce_gain*1000,"o-",label=variant,color=colors[variant]);axes[row,2].plot(h.lr,h.corrected-h.harmed,"o-",label=variant,color=colors[variant])
 for col in [1,2]:
  ax=axes[row,col];ax.axhline(0,color="#555",lw=.8);ax.set_xscale("log");ax.set_xticks([.001,.01,.03,.1],[".001",".01",".03",".1"]);ax.set_xlabel("Single-write learning rate");ax.grid(alpha=.15);ax.legend(frameon=False)
 axes[row,1].set_ylabel("Future CE reduction (× 0.001 nats)");axes[row,2].set_ylabel("Corrected − harmed predictions\n(total across 1,024 future windows)")
for ax,title in zip(axes[0],["Auxiliary task learns input information","Loss changes after a fresh masked write","Classification gains do not follow"]):ax.set_title(title,fontsize=11)
fig.suptitle("A usable auxiliary head is not sufficient for useful test-time writes",fontsize=15,y=.99);fig.text(.5,.012,"Same classifier before/after head fitting. Four reused source-controller participants; 64 reset pulses each, H=4. Exploratory diagnostic.",ha="center",fontsize=9);fig.tight_layout(rect=[0,.05,1,.96]);fig.savefig(r/"phase36/auxiliary_vs_classification.png",dpi=160);plt.close(fig)
p=r/"phase36/report.html";s=p.read_text();s=s.replace("<h1>同一分类器：辅助头训练前后的新鲜写入</h1>","<h1>同一分类器：辅助头训练前后的新鲜写入</h1><img src=\"auxiliary_vs_classification.png\" style=\"max-width:100%\" alt=\"辅助学习、未来损失与分类正确数的对照\">");p.write_text(s)

from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase65";d=pd.read_csv(o/"seed_summary.csv");fig,ax=plt.subplots(figsize=(9,4.6))
for j,(target,color) in enumerate([("current","#3377aa"),("future","#aa3377")]):
 q=d[(d.target==target)&(d.ema==.5)].sort_values("seed");x=np.arange(3)+(j-.5)*.16;ax.scatter(x,q.adaptive_minus_static_pp,color=color,s=65,label=target+" target")
ax.axhline(0,color="black",lw=1);ax.set_xticks(range(3),["17","29","43"]);ax.set_xlabel("Optimizer order seed (both methods; classifier fixed at 17)");ax.set_ylabel("Adaptive minus label-matched static Macro-F1 (pp)");ax.set_title("Repeating both methods preserves the source-side static advantage");ax.legend();ax.grid(axis="y",alpha=.2)
fig.text(.5,.01,"EMA 0.5 for all. Each point averages 4 reused source-validation people; these are not new blind-test results.",ha="center",fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(o/"both_seed_comparison.png",dpi=170);plt.close(fig)
p=o/"report.html";h=p.read_text()
if "both_seed_comparison.png" not in h:p.write_text(h+'<p><img src="both_seed_comparison.png" style="max-width:100%" alt="双方重复优化顺序后与静态对照的差距"></p>')

from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase61";df=pd.read_csv(o/"online_seed_summary.csv")
fig,ax=plt.subplots(figsize=(9,4.6));colors={"current":"#3377aa","future":"#aa3377"}
for j,v in enumerate(["current","future"]):
 q=df[(df.variant==v)&(df.ema==.5)].sort_values("seed")
 x=np.arange(3)+(j-.5)*.12
 ax.plot(x,q.gain_vs_static_pp,"o-",label=v+" target",color=colors[v])
ax.axhline(0,color="black",linewidth=1);ax.set_xticks(range(3),["17 (reused)","29","43"]);ax.set_xlabel("Auxiliary training order seed (classifier fixed at seed 17)");ax.set_ylabel("Macro-F1 difference vs label-matched static (pp)");ax.set_title("Repeated auxiliary training still trails matched static calibration")
ax.grid(axis="y",alpha=.2);ax.legend();fig.text(.5,.01,"Same EMA 0.5. Means over 4 reused source-validation people; each target has its own label-matched static reference.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(o/"seed_replication.png",dpi=170);plt.close(fig)
p=o/"report.html";h=p.read_text()
if "seed_replication.png" not in h:p.write_text(h+'<p><img src="seed_replication.png" style="max-width:100%" alt="固定分类器下辅助头训练顺序种子的重复结果"></p>')

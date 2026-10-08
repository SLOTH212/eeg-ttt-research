from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase27";d=pd.read_csv(o/"seed_means.csv");c=pd.read_csv(o/"paired_comparisons.csv");cohorts=["young_dev","middle_dev","older"];fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,source in zip(axes,["young","mixed"]):
 for seed in [17,29,43]:
  g=d[(d.source==source)&(d.seed==seed)&(d.method=="full_ema08")].set_index("cohort").reindex(cohorts);ax.plot(range(3),g.f1,"o-",label=f"Full encoder, seed {seed}")
 ref=c[(c.source==source)&(c.method=="full_ema08")&(c.reference=="context11_ema_08")].set_index("cohort").reindex(cohorts);ax.plot(range(3),ref.reference_f1,"s--",color="black",label="History baseline, mean 3 seeds")
 ax.set_xticks(range(3),["Young (n=8)","Middle (n=7)","Older (n=19)"]);ax.set_title(source.capitalize()+" source");ax.set_ylim(48,80);ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Macro-F1 (%)");axes[1].legend(fontsize=8);fig.suptitle("Training-seed sensitivity: all methods use causal EMA (alpha=0.8)");fig.tight_layout();fig.savefig(o/"seed_sensitivity.png",dpi=160)
p=o/"report.html";s=p.read_text()
if "seed_sensitivity.png" not in s:s=s.replace("<h2>","<img src='seed_sensitivity.png' style='max-width:100%' alt='Training seed sensitivity'><h2>",1);p.write_text(s)

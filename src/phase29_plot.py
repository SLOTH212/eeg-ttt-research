from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/"phase29";d=pd.read_csv(o/"participants.csv");fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True);people=[7,17,27,37]
for ax,source in zip(axes,["young","mixed"]):
 for offset,weight,color in zip([-.22,0,.22],[0,.01,.1],["#777777","#167ca5","#d78523"]):
  g=d[(d.source==source)&(d.weight==weight)&(d.objective=="masked_token")&(d.lr==.01)].set_index("subject").reindex(people);ax.bar([i+offset for i in range(4)],g.future_ce_gain*1000,width=.21,label=f"Aux weight {weight:g}",color=color)
 ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(4),[f"s{s:02d}" for s in people]);ax.set_title(source.capitalize()+" source");ax.set_xlabel("Source-controller participant");ax.grid(axis="y",alpha=.15)
axes[0].set_ylabel("Future CE reduction (0.001 nats)");axes[1].legend(fontsize=9);fig.suptitle("Independent masked writes: LR=0.01, horizon=4; no online F1 claim");fig.tight_layout();fig.savefig(o/"masked_write_people.png",dpi=160)
p=o/"report.html";s=p.read_text();extra="""<h2>写入是否可观测</h2><p>使用当前熵、置信度、辅助损失、梯度范数、提议预测变化5个特征，按人留一Ridge(alpha100)。未来收益和当前收益分别作为源训练标签；未来信息与真实睡眠标签不作为特征。年轻来源权重0.01、掩码LR0.01：未来5特征策略约0.001125 nats，总是写入0.001121，差异很小；当前收益策略0.000584。未来两项置信特征策略和训练均值策略均等于总是写入。没有确立学习式筛选优势。</p><p><a href="observability_summary.csv">全部可观测性比较</a> · <a href="observability_protocol.json">诊断协议</a></p>"""
if "masked_write_people.png" not in s:s=s.replace("<h1>","<img src='masked_write_people.png' style='max-width:100%' alt='Per-person cold masked write gains'><h1>",1)
if "写入是否可观测" not in s:s=s.replace("</html>",extra+"</html>")
p.write_text(s)

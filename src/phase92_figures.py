from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/"phase92";O.mkdir(exist_ok=True)
assert json.loads((R/"phase91/loaded_head_audit.json").read_text())["constrained_weights_verified_against_beta"]
for phase in [87,88,89,90,91]:assert json.loads((R/f"phase{phase}/audit.json").read_text())["status"]=="PASS"
a=pd.read_csv(R/"phase87/reconstruction_validation.csv");b=pd.read_csv(R/"phase91/reconstruction_validation.csv");s0=pd.read_csv(R/"phase87/summary.csv").set_index(["group","variant"]);s1=pd.read_csv(R/"phase91/summary.csv").set_index(["group","variant"])
labels=["Initial readout","Task-meta","Constrained"];positions=np.arange(3);fig,ax=plt.subplots(2,2,figsize=(14,9))
for sid in [5,15,25,35]:
 values=[float(a[(a.subject==sid)&(a.variant=="initial")].ratio_to_zero.iloc[0]),float(a[(a.subject==sid)&(a.variant=="future")].ratio_to_zero.iloc[0]),float(b[(b.subject==sid)&(b.variant=="future")].ratio_to_zero.iloc[0])]
 ax[0,0].plot(positions,values,"o-",label=f"s{sid:02d}")
ax[0,0].axhline(1,color="#333",ls="--",lw=1);ax[0,0].set_yscale("log");ax[0,0].set_xticks(positions,labels);ax[0,0].set_ylabel("Masked MSE / zero-output MSE (log scale)");ax[0,0].set_title("A. Reconstruction on 4 source-validation people");ax[0,0].legend(ncol=4,fontsize=9);ax[0,0].grid(axis="y",alpha=.2)
for axis,group,title in [(ax[0,1],"controllers","B. Future loss gain: controller people"),(ax[1,0],"validation","C. Future loss gain: source-validation people")]:
 rows=[s0.loc[(group,"initial")],s0.loc[(group,"future")],s1.loc[(group,"future")]]
 for j,(column,color,name) in enumerate([("future_ce_gain","#277da1","Ordinary CE"),("balanced_ce_gain","#f8961e","Class-weighted CE")]):
  values=[float(r[column]) for r in rows];bars=axis.bar(positions+(j-.5)*.32,values,width=.30,color=color,label=name);axis.bar_label(bars,labels=[f"{v:+.5f}" for v in values],fontsize=8,padding=3,rotation=25)
 axis.axhline(0,color="#555",lw=.8);axis.set_xticks(positions,labels);axis.set_ylabel("Mean CE gain (nats; positive is better)");axis.set_title(title);axis.legend(fontsize=9);axis.grid(axis="y",alpha=.2);axis.margins(y=.3)
for j,(group,color) in enumerate([("controllers","#577590"),("validation","#90be6d")]):
 rows=[s0.loc[(group,"initial")],s0.loc[(group,"future")],s1.loc[(group,"future")]];values=[int(r.corrected-r.harmed) for r in rows];bars=ax[1,1].bar(positions+(j-.5)*.34,values,width=.32,color=color,label=group);ax[1,1].bar_label(bars,labels=[f"{v:+d}" for v in values],padding=3)
ax[1,1].axhline(0,color="#555",lw=.8);ax[1,1].set_xticks(positions,labels);ax[1,1].set_ylabel("Corrected minus harmed predictions (counts)");ax[1,1].set_title("D. Classification changes over sampled future windows");ax[1,1].legend(fontsize=9);ax[1,1].margins(y=.25);ax[1,1].grid(axis="y",alpha=.2)
fig.suptitle("CBraMod: useful reconstruction, future loss, and classification are separate tests\nFixed seed17; each group has 4 previously observed source people; cold one-step writes, not online F1",fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(O/"meta_constraint_tradeoff.png",dpi=170);plt.close(fig)
pd.DataFrame([dict(group=g,variant=label,**dict(row)) for g in ["controllers","validation"] for label,row in zip(labels,[s0.loc[(g,"initial")],s0.loc[(g,"future")],s1.loc[(g,"future")]])]).to_csv(O/"figure_task_values.csv",index=False)
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px/1.7 sans-serif;max-width:1400px;margin:35px auto}img{width:100%}</style><h1>辅助重建、后续损失与分类收益不能互相替代</h1><p>原始重拟合辅助头能重建，但写入不利。任务元训练提高后续CE收益，却破坏重建；按源无标签规则把任务头投影回重建有效区域后，平均普通CE仍有微小改善，但另一组源人群的类别加权CE与分类正确性未通过。两条主分支均不推进在线或目标评估。</p><img src="meta_constraint_tradeoff.png"><p>图中Initial readout指用初始有效辅助头做一次写入，不是KEEP。未来损失均相对KEEP；正确性为抽样后续窗口计数，不能转换成整夜Macro-F1。任务头继承1,268个额外源分类标签，约束没有增加分类标签。全部源评估者此前已在项目中观察。本图为事后汇总。</p><p>固定源方向能解释大部分无约束任务头收益；完整记录的固定修正在验证组平均+0.60pp但区间跨0。同标签预算的直接分类头微调也可能在CE改善时损害Macro-F1，进一步说明必须报告真实分类指标和个体伤害。</p><p><a href="../phase87/report.html">无约束结果</a> · <a href="../phase88/report.html">固定方向机制</a> · <a href="../phase89/report.html">完整记录静态对照</a> · <a href="../phase90/report.html">源解析约束</a> · <a href="../phase91/report.html">约束后正式复核及加载事件</a></p>')
(O/"audit.json").write_text(json.dumps(dict(status="PASS",source_phases=[87,88,89,90,91],corrected_phase91_loaded_head_verified=True,retrospective_visual_summary=True,no_new_analysis_selection=True),indent=2));print("Scientific figure and report complete")

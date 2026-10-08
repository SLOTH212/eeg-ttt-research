from common import *
from phase2_temporal import ema
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
o=ROOT/"phase70";protocol=json.loads((o/"protocol.json").read_text());names=["W","N1","N2","N3","REM"];rows=[];people=[]
for sid in protocol["subjects"]:
 z=np.load(o/f"s{sid:02d}.npz");ref=np.load(ROOT/"phase40"/f"s{sid:02d}.npz");y=z["y"];valid=y>=0;truth=y[valid];a=ema(z["pretrained"],.5)[valid].argmax(1);b=ema(ref["frozen"],.5)[valid].argmax(1)
 ca=confusion_matrix(truth,a,labels=range(5));cb=confusion_matrix(truth,b,labels=range(5));f1a=2*ca.diagonal()/np.maximum(ca.sum(0)+ca.sum(1),1);f1b=2*cb.diagonal()/np.maximum(cb.sum(0)+cb.sum(1),1);delta=100*(f1a-f1b)
 for stage in range(5):
  rows.append(dict(subject=sid,cohort="young" if sid<20 else "middle" if sid<40 else "older",stage=names[stage],true_windows=int(ca[stage].sum()),true_present=bool(ca[stage].sum()),cbramod_f1=100*f1a[stage],labram_f1=100*f1b[stage],f1_difference_pp=delta[stage],cbramod_recall=100*ca[stage,stage]/max(ca[stage].sum(),1),labram_recall=100*cb[stage,stage]/max(cb[stage].sum(),1),recall_difference_pp=100*(ca[stage,stage]-cb[stage,stage])/max(ca[stage].sum(),1),corrected=int(((a==truth)&(b!=truth)&(truth==stage)).sum()),harmed=int(((a!=truth)&(b==truth)&(truth==stage)).sum())))
 original=pd.read_csv(o/"participants.csv");g=original[(original.subject==sid)&(original.ema==.5)].set_index("model");assert abs(delta.mean()-(g.loc["pretrained","macro_f1"]-g.loc["labram","macro_f1"]))<1e-10
 people.append(dict(subject=sid,macro_f1_difference_pp=delta.mean(),accuracy_difference_pp=100*((a==truth).mean()-(b==truth).mean()),largest_negative_stage=names[int(delta.argmin())],largest_positive_stage=names[int(delta.argmax())]))
df=pd.DataFrame(rows);df.to_csv(o/"stage_people.csv",index=False);summary=df.groupby("stage").agg(cbramod_f1=("cbramod_f1","mean"),labram_f1=("labram_f1","mean"),f1_difference_pp=("f1_difference_pp","mean"),recall_difference_pp=("recall_difference_pp","mean"),people_with_true_stage=("true_present","sum"),corrected=("corrected","sum"),harmed=("harmed","sum")).reindex(names);summary.to_csv(o/"stage_summary.csv")
ppl=pd.DataFrame(people).sort_values("macro_f1_difference_pp");ppl.to_csv(o/"negative_transfer_people.csv",index=False)
mat=df.pivot(index="subject",columns="stage",values="f1_difference_pp").reindex(index=ppl.subject,columns=names)
fig,axes=plt.subplots(1,2,figsize=(11,8),gridspec_kw={"width_ratios":[2,1]})
limit=max(10.,float(abs(mat.to_numpy()).max()));im=axes[0].imshow(mat.to_numpy(),aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit);axes[0].set_xticks(range(5),names);axes[0].set_yticks(range(30),[f"s{s:02d}" for s in mat.index]);axes[0].set_title("Per-stage F1 difference (pp)")
axes[1].barh(range(30),ppl.macro_f1_difference_pp,color=np.where(ppl.macro_f1_difference_pp>=0,"#3377aa","#bb5566"));axes[1].set_ylim(len(ppl)-.5,-.5);axes[1].set_yticks([]);axes[1].axvline(0,color="black",lw=.8);axes[1].set_title("Overall Macro-F1 difference (pp)");axes[1].grid(axis="x",alpha=.2)
fig.colorbar(im,ax=axes[0],shrink=.6);fig.suptitle("CBraMod vs LaBraM: heterogeneous static transfer (EMA 0.5)")
fig.text(.5,.01,"30 previously observed target people, one classifier seed. Stage differences describe errors; they do not establish causes.",ha="center",fontsize=9);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(o/"stage_transfer.png",dpi=170);plt.close(fig)
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1150px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>静态迁移的阶段得失与个体损害</h1><p>事后描述分析，固定阶段70主比较：CBraMod预训练与LaBraM同EMA.5。每个阶段先按人计算F1，再平均；不存在真标签的阶段仍按固定五类指标记0，同时列出有该阶段的受试者数。正确/受损窗口数为总计，与按人均值的加权方式不同。</p>'+summary.to_html(float_format=lambda x:f"{x:.4f}")+'<p><img src="stage_transfer.png" style="max-width:100%" alt="逐受试者、逐睡眠阶段的迁移差异"></p>'+ppl.to_html(index=False,float_format=lambda x:f"{x:.4f}")+'<p>分阶段F1差值的平均精确等于每人的Macro-F1差值。阶段分解帮助定位错误类型，不证明变化的生理或模型因果原因，不用于改动已经固定的目标分类器。</p></html>';(o/"stage_analysis.html").write_text(html);p=o/"report.html";h=p.read_text()
if "stage_analysis.html" not in h:p.write_text(h+'<p><a href="stage_analysis.html">事后阶段与个体损害分解</a></p>')
print(summary.to_string());print(ppl.head(5).to_string(index=False))

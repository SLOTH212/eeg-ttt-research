from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase108";O.mkdir(exist_ok=True);pr=json.loads((ROOT/"phase107/protocol.json").read_text());variants=["relative_repeat11","relative_context11"];rows=[];audits=[]
protocol=dict(purpose="Retrospective descriptive transition-context diagnosis of phase107 relativehistory failure; no fitting, selection, or onlinegate.",parent_protocol_sha256=hashlib.sha256((ROOT/"phase107/protocol.json").read_bytes()).hexdigest(),strata="Scored currentwindows: startup t<10; history_unscored ifanyprevious10labels<0; otherwise transition_context ifcurrent+previous10labels not allsame; stable_context ifallsame.",metrics="Withinperson3seed meanaccuracy and CE as mainmechanism descriptors; macroF1 separatelydescriptive withfixedfiveclasses. Strata labels unavailableonline, never usedforprediction.",variants=variants,ema=[1.,.5],scope="Samealreadyobserved92cases; allstratafixedbeforethis diagnostic scoring. No newblindclaim or correctivehyperparameters.")
(O/"protocol.json").write_text(json.dumps(protocol,indent=2))
for c in pr["cases"]:
 name="_".join(c[k] for k in ["domain","subject","montage"]);z=np.load(ROOT/"phase107/predictions"/(name+".npz"));yy=np.load(c["labels"]);y=yy["y"] if c["labels"].endswith(".npz") else yy;strata=np.full(len(y),"unscored",dtype="U24")
 for t in np.flatnonzero(y>=0):
  strata[t]="startup" if t<10 else "history_unscored" if (y[t-10:t]<0).any() else "stable_context" if (y[t-10:t+1]==y[t]).all() else "transition_context"
 assert int((strata!="unscored").sum())==int((y>=0).sum());counts={stratum:int((strata==stratum).sum()) for stratum in ["startup","history_unscored","stable_context","transition_context"]};audits.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],scored=int((y>=0).sum()),counts=counts))
 for variant in variants:
  for seed in [17,29,43]:
   p=z[f"{variant}_seed{seed}"]
   for alpha in protocol["ema"]:
    q=p if alpha==1 else ema(p,alpha)
    for stratum,n in counts.items():
     if not n:continue
     mask=strata==stratum;mm=metrics(y[mask],q[mask]);cm=np.array(mm["confusion"]);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],variant=variant,seed=seed,ema=alpha,stratum=stratum,n=n,accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"],macro_f1=100*mm["macro_f1"]))
df=pd.DataFrame(rows);df.to_csv(O/"participants.csv",index=False);keys=["domain","subject","montage","ema","stratum","variant","n"];person=df.groupby(keys)[["accuracy","ce","macro_f1"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);deltas=[]
for (domain,sid,montage,alpha,stratum,n),g in person.groupby(["domain","subject","montage","ema","stratum","n"]):
 g=g.set_index("variant");a=g.loc["relative_context11"];b=g.loc["relative_repeat11"];deltas.append(dict(domain=domain,subject=sid,montage=montage,ema=alpha,stratum=stratum,n=n,accuracy_gain=a.accuracy-b.accuracy,ce_gain=b.ce-a.ce,macro_f1_gain=a.macro_f1-b.macro_f1,history_accuracy=a.accuracy,repeat_accuracy=b.accuracy))
dd=pd.DataFrame(deltas);dd.to_csv(O/"paired_people.csv",index=False);summary=dd.groupby(["domain","montage","ema","stratum"]).agg(people=("subject","nunique"),scored_windows=("n","sum"),accuracy_gain=("accuracy_gain","mean"),ce_gain=("ce_gain","mean"),macro_f1_gain=("macro_f1_gain","mean"),history_accuracy=("history_accuracy","mean"),repeat_accuracy=("repeat_accuracy","mean")).reset_index();summary.to_csv(O/"summary.csv",index=False)
# Check that stratum weights recover whole-record accuracy/CE differences exactly.
whole=pd.read_csv(ROOT/"phase107/seedmean_people.csv");errs=[]
for (domain,sid,montage,alpha),g in dd.groupby(["domain","subject","montage","ema"]):
 w=whole[(whole.domain==domain)&(whole.subject==sid)&(whole.montage==montage)&(whole.ema==alpha)].set_index("variant");a=w.loc["relative_context11"];b=w.loc["relative_repeat11"];ae=abs(np.average(g.accuracy_gain,weights=g.n)-(a.accuracy-b.accuracy));ce=abs(np.average(g.ce_gain,weights=g.n)-(b.ce-a.ce));tol=8*np.finfo(np.float32).eps*max(1.,abs(a.ce),abs(b.ce));assert ae<1e-10 and ce<tol;errs.append([ae,ce])
audit=dict(status="PASS",cases=len(audits),strata_partition_scored_windows=True,weighted_accuracy_recovery_exact_to_1e10=True,weighted_ce_recovery_within_float32_rounding=True,max_accuracy_recovery_error=float(np.array(errs)[:,0].max()),max_ce_recovery_error=float(np.array(errs)[:,1].max()),ce_tolerance="8*float32epsilon*max(1,abs(eachwholeCE)); groupmean rounding, predictions unchanged",no_prediction_changes=True,no_label_based_gate=True,records=audits)
(O/"audit.json").write_text(json.dumps(audit,indent=2))
display=summary[summary.montage.isin(["native","left"])]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>相对频谱历史失败：阶段切换与稳定片段诊断</h1><p>保留phase107全部预测，只按当前及过去10窗的真实标签分层：开始不足10窗、历史存在未评分窗、最近11窗同阶段、最近11窗有阶段变化。标签只用于事后解释，实际输入仍完全无标签；未据此设计门控或重新训练。EMA1和0.5均固定报告。</p>'+display.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>正accuracy_gain表示相对历史模型优于同维度重复当前窗；正CEgain表示损失改善。各人先平均三个种子，再取人平均。分层人数可能不同；Macro-F1在阶段缺失的分层中只作辅助描述，不能相加还原总体值。按窗数加权的accuracy差异还原至1e-10内；CE分组均值受float32舍入影响，已按数值精度界限复核，最大误差记录在审计中。</p><p>关联分层不能证明失败原因，也不能把稳定片段标签用于部署时选择更新。阶段持续时间、年龄、导联和其他分布变化仍可能共同作用。</p><p><a href="audit.json">分层完备与指标还原审计</a> · <a href="paired_people.csv">每位受试者分层结果</a> · <a href="../phase107/report.html">保留原主比较</a></p>')
print(display.to_string(index=False));print({k:v for k,v in audit.items() if k!="records"})

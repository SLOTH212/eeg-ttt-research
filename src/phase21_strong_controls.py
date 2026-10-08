from common import *
from phase2_temporal import ema
import pandas as pd
out=ROOT/"phase21";rows=[]
for f in sorted((out/"evaluation").glob("*/evaluation.json")):
 d=json.loads(f.read_text())
 if d["kind"]!="pretrained":continue
 for r in d["results"]:
  sid=r["subject"];cache=np.load(f.parent/f"s{sid:02d}.npz");y=cache["y"];p=cache["source_tuned"];mask=np.zeros(len(y),bool)
  for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
  for method,pr in [("full_source_tuned",p),("full_source_tuned_ema08",ema(p,.8))]:
   rows.append(dict(source=d["source"],subject=sid,cohort=r["cohort"],method=method,f1=metrics(y,pr)["macro_f1"],transition=metrics(y[mask],pr[mask])["macro_f1"]))
frame=pd.DataFrame(rows);frame.to_csv(out/"strong_control_participants.csv",index=False)
a=pd.read_csv(ROOT/"phase4/all_metrics.csv");a["cohort"]=a.cohort.map({"young":"young_dev","middle":"middle_dev"});b=pd.read_csv(ROOT/"phase6/all_metrics.csv");b["cohort"]="older";old=pd.concat([a,b],ignore_index=True);comp=[]
for keys,g in frame.groupby(["source","cohort","method"]):
 for variant in ["full52","wide","context11"]:
  for method in ["frozen","ema_08"]:
   ref=old[(old.source==keys[0])&(old.cohort==keys[1])&(old.variant==variant)&(old.method==method)]
   for seeds in ["seed17","mean3seeds"]:
    if seeds=="seed17":ref2=ref[ref.seed==17]
    else:ref2=ref
    reference=ref2.groupby("subject")[["f1","transition"]].mean();q=g.set_index("subject")[["f1","transition"]].join(reference,rsuffix="_reference",how="inner");assert len(q)==len(g)
    delta=100*(q.f1-q.f1_reference).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);comp.append(dict(source=keys[0],cohort=keys[1],method=keys[2],reference=variant+"_"+method,reference_seeds=seeds,n=len(q),carrier_f1=100*q.f1.mean(),reference_f1=100*q.f1_reference.mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],transition_delta_pp=100*(q.transition-q.transition_reference).mean(),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min()))
summary=pd.DataFrame(comp);summary.to_csv(out/"strong_control_comparisons.csv",index=False);selected=summary[(summary.reference=="context11_ema_08")&(summary.reference_seeds=="seed17")]
(out/"strong_controls.html").write_text("<!doctype html><html lang=zh><meta charset=utf-8><title>全编码器与既有强基线</title><style>body{max-width:1400px;margin:35px auto;font:15px/1.7 system-ui}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>与既有强基线的同人配对比较</h1><p>LaBraM完整源域训练的当前窗口预测及固定alpha=.8因果平滑，对照此前52维频谱、加宽网络、11窗口历史模型。来源和评估人员匹配；下表对照seed17，CSV另提供此前三种子均值。模型的来源验证调参预算不同，当前仅一个种子。11窗口基线可以使用过去5分钟，而LaBraM编码器只使用当前30秒，因此该比较衡量实际可用表现，不隔离预训练贡献。所有结果均为探索性，不能覆盖早期失败的锁定主检验。</p>"+selected.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<p>transition_delta_pp为人工标签标记的阶段转换前后窗口上的离线F1差异；转换标签不用于预测。</p><a href=strong_control_comparisons.csv>所有预定义比较</a> · <a href=report.html>完整训练报告</a></html>")
print(selected.to_string(index=False))

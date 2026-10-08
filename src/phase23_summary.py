from common import *
import pandas as pd
out=ROOT/"phase23";protocol=json.loads((out/"protocol.json").read_text());rows=[];cases=0;audited=0;max_error=0.
for f in sorted(out.glob("*/s*.json")):
 d=json.loads(f.read_text());sid=d["subject"];assert sid in protocol["evaluation"] and sid not in protocol["reserved_controllers"];saved=np.load(f.with_suffix(".npz"));y=saved["y"];data=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(y,np.load(data/f"s{sid:02d}_meta.npz")["y"]);mask=np.zeros(len(y),bool)
 for b in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1:mask[max(0,b-2):min(len(y),b+3)]=True
 assert d["nonwritable_state_unchanged"];cases+=1
 for r in d["results"]:
  pr=saved[r["method"]];assert pr.shape==(len(y),5) and np.isfinite(pr).all() and np.max(np.abs(pr.sum(1)-1))<2e-6;m=metrics(y,pr);transition=metrics(y[mask],pr[mask])["macro_f1"]
  for k in ["macro_f1","ce","balanced_accuracy"]:max_error=max(max_error,abs(m[k]-r[k]))
  max_error=max(max_error,abs(transition-r["transition_f1"]));audited+=1;rows.append(dict(source=d["source"],objective=d["objective"],subject=sid,cohort=d["cohort"],method=r["method"],f1=m["macro_f1"],transition=transition,ce=m["ce"],lr=d["lr"]))
if not rows:raise SystemExit("No completed targets")
frame=pd.DataFrame(rows);frame.to_csv(out/"participant_metrics.csv",index=False);complete=[];expected={"young_dev":6,"middle_dev":5,"older":19}
for keys,g in frame.groupby(["source","objective","cohort"]):
 if g.subject.nunique()==expected[keys[2]]:complete.append(g)
for keys,g in frame.groupby(["source","objective"]):
 if g.subject.nunique()==30:q=g.copy();q["cohort"]="all30";complete.append(q)
summ=[];paired=[]
for g in complete:
 source=g.source.iloc[0];obj=g.objective.iloc[0];cohort=g.cohort.iloc[0]
 for method,h in g.groupby("method"):summ.append(dict(source=source,objective=obj,cohort=cohort,method=method,n=h.subject.nunique(),f1=100*h.f1.mean(),transition_f1=100*h.transition.mean(),ce=h.ce.mean()))
 pivot=g.pivot(index="subject",columns="method",values="f1");transition=g.pivot(index="subject",columns="method",values="transition")
 for baseline in ["frozen","weak_ema","strong_ema"]:
  delta=100*(pivot.source_selected-pivot[baseline]).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);paired.append(dict(source=source,objective=obj,cohort=cohort,baseline=baseline,n=len(delta),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],wins=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min(),transition_delta_pp=100*(transition.source_selected-transition[baseline]).mean()))
s=pd.DataFrame(summ);q=pd.DataFrame(paired);s.to_csv(out/"summary.csv",index=False);q.to_csv(out/"paired.csv",index=False);assert max_error<1e-10
(out/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",completed_person_cases=cases,expected_person_cases=120,saved_prediction_streams=audited,max_metric_error=max_error,target_labels_match_original=True,controller_people_excluded=True,note="Saved prediction audit plus runtime nonwritable-state assertions; repeated methods are not independent evidence. Only complete cohort groups reported."),indent=2))
selection=json.loads((out/"selection_lock.json").read_text());show=q[(q.objective=="entropy")&(q.baseline=="weak_ema")] if len(q) else q
html="<!doctype html><html lang=zh><meta charset=utf-8><title>完整训练模型的连续适应评估</title><style>body{max-width:1300px;margin:40px auto;font:16px/1.7 system-ui}td,th{padding:6px;border-bottom:1px solid #ddd}</style><h1>完整训练模型：源域选择后的连续适应</h1><p>完成 "+str(cases)+" /120个人配置。仅展示人员齐全的人群分组；每种来源30位评估人员（年轻6、中年5、老年19），另4位控制器人员排除。所有人群曾在此前探索中观察过，不是新盲测。一个训练种子。source_selected参数仅由phase22的四位源验证人员选定，目标标签仅用于评分。</p><p>两种来源的熵更新均选lr=.001；时间教师均回退冻结。下面主对照为熵更新减去固定alpha=.8简单平滑，区间按人配对bootstrap95%。</p>"+show.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<h2>全部方法均值</h2>"+s.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<p>harm_over_1pp为下降超过1个百分点的人数。转换窗口用人工阶段标签离线标记，标签不参与更新。时间教师回退冻结的重复结果不构成额外独立证据。</p><p><a href=selection_lock.json>目标评估前保存的选择与模型哈希</a> · <a href=paired.csv>全部配对比较</a> · <a href=prediction_audit.json>预测审计</a> · <a href=../phase22/report.html>源验证完整筛选</a></p></html>"
(out/"report.html").write_text(html);print(show.to_string(index=False));print("AUDIT",cases,audited,max_error)

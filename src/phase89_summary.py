from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase89";assert (O/"complete.json").exists();pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();assert json.loads((O/"execution_audit.json").read_text())["status"]=="PASS";ta=json.loads((O/"training_audit.json").read_text());assert ta["status"]=="PASS"
for which in ["current","future"]:
 used=ta["records"][which]["labels_used"];meta=json.loads((ROOT/"phase86"/which/"complete.json").read_text());ref=sorted([[int(s),int(t)] for s,ts in meta["labels_used"].items() for t in ts]);assert sorted(used)==ref;assert ta["records"][which]["label_presentations"]==meta["label_count"]*5;assert ta["records"][which]["optimizer_steps"]==400
rows=[]
for sid in pr["subjects"]:
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];d=np.load(O/"predictions"/f"s{sid:02d}.npz");assert set(d.files)==set(pr["methods"])
 for method in pr["methods"]:
  for alpha in pr["ema"]:
   p=d[method] if alpha==1 else ema(d[method],alpha);m=metrics(y,p);cm=np.array(m["confusion"]);rows.append(dict(subject=sid,group="validation" if sid in pr["primary_people"] else "controllers",method=method,ema=alpha,macro_f1=100*m["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=m["ce"]))
df=pd.DataFrame(rows);assert len(df)==120;df.to_csv(O/"participants.csv",index=False);summary=df.groupby(["group","method","ema"])[["macro_f1","accuracy","ce"]].mean();summary.to_csv(O/"summary.csv");comparisons=[];rng=np.random.default_rng(89017)
for (group,alpha),g in df.groupby(["group","ema"]):
 p=g.pivot(index="subject",columns="method",values="macro_f1");ce=g.pivot(index="subject",columns="method",values="ce");idx=rng.integers(0,4,(10000,4))
 for method in pr["methods"][1:]:
  v=p[method]-p.frozen;ci=np.quantile(v.to_numpy()[idx].mean(1),[.025,.975]);comparisons.append(dict(group=group,ema=alpha,method=method,frozen_f1=p.frozen.mean(),adapted_f1=p[method].mean(),delta_pp=v.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((v>0).sum()),harm_over_1pp=int((v< -1).sum()),worst_pp=v.min(),ce_gain=(ce.frozen-ce[method]).mean()))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"paired_comparisons.csv",index=False)
audit=dict(status="PASS",method_streams_recomputed=120,people=8,classifier_seed=17,exact_same_meta_label_indices=True,exact_same_label_presentations=True,optimizer_steps_perhead=400,parameters_static_head=2005,parameters_meta_auxiliary_head=40200,compute_and_parameter_roles_notmatched=True,no_new_target_evaluation=True,all_source_people_previously_seen=True,all_corrections_fixed_before_record=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>CBraMod同标签预算静态对照：完整源记录</h1><p>仅使用既有seed17源分类器，八位源评估者均未参与新增五位辅助训练，但此前项目已观察。固定源均值参数偏移在记录开始前一次性应用；直接分类头对照用相同317/1,268个标签及五轮标签呈现训练，400次AdamW更新，无源验证选轮、学习率搜索或目标调参。静态分类头2,005参数，元辅助头40,200参数，作用位置与计算成本不同，不能声称全计算预算匹配。</p><p>本阶段评估完整记录的Macro-F1，避免把稀疏脉冲的CE改善误认为整夜分类收益。没有对此前未通过门槛的动态写入做在线或目标推广。</p>'+comp.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>逐人同EMA0.5结果</h2>'+df[df.ema==.5].to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p>每组只有4人，区间为固定seed17模型下描述性受试者bootstrap，不涵盖全部选择不确定性。不同分类标签预算不能混成一个公平比较：current对current，future对future。</p><p><a href="audit.json">标签与指标审计</a> · <a href="training_audit.json">训练细节</a> · <a href="../phase87/report.html">保留的动态写入失败</a> · <a href="../phase88/report.html">固定方向机制分析</a></p>')
print(comp[comp.ema==.5].to_string(index=False));print(audit)

from common import *
import pandas as pd,hashlib
O=ROOT/"phase103";pr=json.loads((O/"protocol.json").read_text());assert json.loads((O/"audit.json").read_text())["status"]=="PASS";weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);rows=[];checks=[]
for group,sids in [("controllers",pr["controllers"]),("validation",pr["validation"])]:
 for sid in sids:
  source=json.loads((O/f"s{sid:02d}.json").read_text());assert source["state_restored_exact"]
  for r in source["rows"]:
   y=np.array(r["future_labels"]);b=np.array(r["future_base_probabilities"]);w=np.array(r["future_write_probabilities"]);assert (b>0).all() and (w>0).all();g=np.log(w[np.arange(4),y])-np.log(b[np.arange(4),y]);err=abs(g.mean()-r["future_ce_gain"]);assert err<1e-12 and abs((g*weights[y]).mean()-r["balanced_ce_gain"])<1e-12;assert int(((b.argmax(1)!=y)&(w.argmax(1)==y)).sum())==r["corrected"] and int(((b.argmax(1)==y)&(w.argmax(1)!=y)).sum())==r["harmed"];checks.append(err);rows.append(r)
  for phase,mapping in [("phase91",{"future":"constrained_wave","initial":"initial_wave"}),("phase87",{"future":"unconstrained_wave_diagnostic"})]:
   for r in json.loads((ROOT/phase/f"s{sid:02d}.json").read_text())["rows"]:
    if r["variant"] in mapping:
     rr=r.copy();rr["variant"]=mapping[r["variant"]];rows.append(rr)
df=pd.DataFrame(rows);keys=["group","subject","variant"];person=df.groupby(keys).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),current_ce_gain=("current_ce_gain","mean"),actual_update_norm=("actual_update_norm","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),argmax_changes=("argmax_changes","sum")).reset_index();person.to_csv(O/"people.csv",index=False)
summary=person.groupby(["group","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),current_ce_gain=("current_ce_gain","mean"),actual_update_norm=("actual_update_norm","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())),corrected_occurrences=("corrected","sum"),harmed_occurrences=("harmed","sum"),argmax_change_occurrences=("argmax_changes","sum")).reset_index();summary.to_csv(O/"summary.csv",index=False)
pairs=[]
for group,g in person.groupby("group"):
 p=g.pivot(index="subject",columns="variant",values="future_ce_gain")
 for mode in ["oracle_current_matched","oracle_future_matched","oracle_current_clip","oracle_future_clip"]:
  d=p[mode]-p.constrained_wave;pairs.append(dict(group=group,variant=mode,mean_extra_future_ce_gain=d.mean(),positive_people=int((d>0).sum()),people=len(d)))
pd.DataFrame(pairs).to_csv(O/"paired_differences.csv",index=False)
audit=dict(status="PASS",new_pulse_rows=2048,compared_pulse_rows=len(df),independent_log_probability_ce_error=max(checks),per_person_means=True,occurrence_counts_may_overlap_not_unique_future_samples=True,labels_used_for_oracle_disclosed=True,no_online_or_target_adaptation=True,protocol_sha256=hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest())
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>单次WRITE的方向与幅度：源数据监督诊断</h1><p>沿用既有8位源受试者、每人64个时点和9600个LN参数。oracle_current使用当前真实标签；oracle_future使用未来四窗标签，是明确的非因果诊断。clip沿用0.01学习率和梯度范数上限1；matched逐脉冲匹配受重建约束的未来元学习头在phase91中的实际更新范数。每个脉冲重置至固定分类器。</p><p>这些梯度不是无标签可部署方法。未来标签对照只是局部优化正对照，不是全局可达上界。匹配范数旨在区分方向与幅度，不建立在线分类收益。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>同受试者对照</h2>'+pd.DataFrame(pairs).to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<p>正CE增益代表损失下降；修正/伤害计数按脉冲未来窗出现次数统计，可能重叠，不能相加成整夜准确率或Macro-F1。无约束wave分支已失败重建有效性，只作机制参照；受约束分支已有失败结论保留。本实验不触发目标集或在线参数更新。</p><p><a href="audit.json">参数与范数审计</a> · <a href="summary_audit.json">独立概率重算</a> · <a href="people.csv">按人结果</a></p>')
print(summary.to_string(index=False));print(audit)

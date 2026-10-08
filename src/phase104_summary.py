from common import *
import pandas as pd,hashlib
O=ROOT/"phase104";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();assert json.loads((O/"audit.json").read_text())["status"]=="PASS";weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);rows=[];errors=[]
for group,sids in [("controllers",pr["controllers"]),("validation",pr["validation"])]:
 for sid in sids:
  data=json.loads((O/f"s{sid:02d}.json").read_text());assert data["state_restored_exact"] and data["future_input_cannot_change_current_student_logit"]
  for r in data["rows"]:
   y=np.array(r["future_labels"]);b=np.array(r["future_base_probabilities"]);w=np.array(r["future_write_probabilities"]);tp=np.array(r["future_teacher_probabilities"]);assert (w>0).all() and (tp>0).all()
   candidates={r["variant"]:w}
   if r["variant"]=="teacher_soft_clip":candidates.update(keep=b,direct_teacher=tp,static_half_blend=.5*b+.5*tp)
   for variant,q in candidates.items():
    delta=np.log(q[np.arange(4),y])-np.log(b[np.arange(4),y]);corrected=int(((b.argmax(1)!=y)&(q.argmax(1)==y)).sum());harmed=int(((b.argmax(1)==y)&(q.argmax(1)!=y)).sum())
    if variant==r["variant"]:
     err=abs(delta.mean()-r["future_ce_gain"]);assert err<1e-12 and abs((delta*weights[y]).mean()-r["balanced_ce_gain"])<1e-12 and corrected==r["corrected"] and harmed==r["harmed"];errors.append(err)
    rows.append(dict(group=group,subject=sid,t=r["t"],variant=variant,future_ce_gain=delta.mean(),balanced_ce_gain=(delta*weights[y]).mean(),accuracy=100*(q.argmax(1)==y).mean(),corrected=corrected,harmed=harmed,current_teacher_correct=r["current_teacher_correct"],actual_update_norm=r["actual_update_norm"] if variant==r["variant"] else 0.))
df=pd.DataFrame(rows);assert len(df)==3072;df.to_csv(O/"pulses.csv",index=False);person=df.groupby(["group","subject","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),actual_update_norm=("actual_update_norm","mean")).reset_index();person.to_csv(O/"people.csv",index=False)
summary=person.groupby(["group","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())),corrected=("corrected","sum"),harmed=("harmed","sum"),actual_update_norm=("actual_update_norm","mean")).reset_index();summary.to_csv(O/"summary.csv",index=False)
gates=[]
for group in ["controllers","validation"]:
 tab=summary[summary.group==group].set_index("variant");r=tab.loc["teacher_soft_clip"];useful=bool(r.future_ce_gain>0 and r.positive_people>=3 and r.balanced_ce_gain>=0 and r.corrected>=r.harmed and r.corrected>0)
 competitive=all(r.future_ce_gain>=tab.loc[v,"future_ce_gain"] and r.accuracy>=tab.loc[v,"accuracy"] for v in ["direct_teacher","static_half_blend"]);gates.append(dict(group=group,useful_write=useful,competitive_with_static=bool(competitive)))
decision=dict(useful_write_all_groups=all(x["useful_write"] for x in gates),advance_online=all(x["useful_write"] and x["competitive_with_static"] for x in gates),groups=gates,primary="teacher_soft_clip",no_secondary_replaces_primary=True)
(O/"decision.json").write_text(json.dumps(decision,indent=2))
conditional=df[df.variant=="teacher_soft_clip"].groupby(["group","current_teacher_correct"]).agg(pulses=("t","size"),future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index();conditional.to_csv(O/"teacher_error_diagnostic.csv",index=False)
audit=dict(status="PASS",new_write_rows=1536,with_static_rows=len(df),independent_ce_error=max(errors),same_teacher_static_controls=True,future_teacher_only_evaluation_not_write=True,occurrence_counts_not_unique_windows=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>固定教师信号能否产生有用WRITE？</h1><p>固定context11 seed17为教师，CBraMod seed17为学生。教师已用相同10位源训练者训练，但有额外模型计算和当前加10窗历史；这些信息优势不能算作TTT本身的收益。当前教师概率产生soft交叉熵，0.01学习率、梯度范数1、9600个LN参数，仅更新一次后评价后续四窗；每个脉冲重置。hard与匹配旧wave更新范数仅为消融。</p><p>未来教师输出只用于相应窗口到达时的静态对照与评分，不进入当前参数更新。与直接教师、固定0.5概率平均同时比较。全部源受试者此前已观察，不是新的盲验证。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>预定主分支判断</h2><pre>'+json.dumps(decision,indent=2)+'</pre><h2>教师当前预测错误的诊断分层</h2>'+conditional.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<p>分层仅用于解释，未用于选择时点或门控。修正/伤害按可能重叠的未来窗出现次数统计，不是整夜Macro-F1。原定soft主分支不能用更好的消融替代。初次验证读取遇到历史预测存储路径缺失，已归档并修正到同检查点的phase79原始输入预测；没有变更科学协议或有效控制组记录。</p><p><a href="protocol.json">固定协议</a> · <a href="summary_audit.json">独立统计审计</a> · <a href="legacy_path_incident.json">技术修正记录</a></p>')
print(summary.to_string(index=False));print(decision);print(audit)

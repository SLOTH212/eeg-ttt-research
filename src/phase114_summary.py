from common import *
import pandas as pd, hashlib
O=ROOT/"phase114"; weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"])
assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip()
rows=[]; errors=[]
for seed in [17,29,43]:
 for group,sids in [("controllers",[7,17,27,37]),("validation",[5,15,25,35])]:
  for sid in sids:
   path=ROOT/"phase104"/f"s{sid:02d}.json" if seed==17 else O/f"seed{seed}"/f"s{sid:02d}.json"
   data=json.loads(path.read_text())
   selected=[r for r in data["rows"] if r["variant"]=="teacher_soft_clip"]
   assert len(selected)==64
   for r in selected:
    y=np.array(r["future_labels"]); b=np.array(r["future_base_probabilities"]); w=np.array(r["future_write_probabilities"]); tp=np.array(r["future_teacher_probabilities"])
    for variant,q in dict(keep=b,teacher_soft_clip=w,direct_teacher=tp,static_half_blend=.5*b+.5*tp).items():
     assert q.shape==(4,5) and (q>0).all() and np.allclose(q.sum(1),1,atol=1e-12,rtol=0)
     delta=np.log(q[np.arange(4),y])-np.log(b[np.arange(4),y])
     corrected=int(((b.argmax(1)!=y)&(q.argmax(1)==y)).sum()); harmed=int(((b.argmax(1)==y)&(q.argmax(1)!=y)).sum())
     if variant=="teacher_soft_clip":
      e=abs(delta.mean()-r["future_ce_gain"]); errors.append(e)
      assert e<1e-12 and abs((delta*weights[y]).mean()-r["balanced_ce_gain"])<1e-12
      assert corrected==r["corrected"] and harmed==r["harmed"]
     rows.append(dict(seed=seed,group=group,subject=sid,t=r["t"],variant=variant,future_ce_gain=delta.mean(),balanced_ce_gain=(delta*weights[y]).mean(),accuracy=100*np.mean(q.argmax(1)==y),corrected=corrected,harmed=harmed))
df=pd.DataFrame(rows); assert len(df)==6144; df.to_csv(O/"pulses.csv",index=False)
person_seed=df.groupby(["seed","group","subject","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index()
person_seed.to_csv(O/"people_seeds.csv",index=False)
person=person_seed.groupby(["group","subject","variant"])[["future_ce_gain","balanced_ce_gain","accuracy","corrected","harmed"]].mean().reset_index(); person.to_csv(O/"people.csv",index=False)
def summarize(p,keys):
 return p.groupby(keys).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index()
summary=summarize(person,["group","variant"]); summary.to_csv(O/"summary.csv",index=False)
per_seed=summarize(person_seed,["seed","group","variant"]);per_seed.to_csv(O/"seed_summary.csv",index=False)
gates=[]
for group in ["controllers","validation"]:
 tab=summary[summary.group==group].set_index("variant"); r=tab.loc["teacher_soft_clip"]
 gates.append(dict(group=group,useful_fresh_write=bool(r.future_ce_gain>0 and r.positive_people>=3 and r.balanced_ce_gain>=0 and r.corrected>=r.harmed and r.corrected>0),competitive_with_static=bool(all(r.future_ce_gain>=tab.loc[v,"future_ce_gain"] and r.accuracy>=tab.loc[v,"accuracy"] for v in ["direct_teacher","static_half_blend"]))))
decision=dict(groups=gates,advance_online=False,advance_target=False,reason="Prespecified source robustness characterization only; does not reopen phase104 failed competitive gate.")
(O/"decision.json").write_text(json.dumps(decision,indent=2))
audit=dict(status="PASS",write_pulses=1536,new_write_pulses=1024,with_static_rows=len(df),max_independent_ce_error=max(errors),aggregation="mean seeds within person; no probability ensemble or seed pseudoreplication",corrected_harmed="seed-averaged counts of potentially overlapping future-window occurrences")
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>教师驱动单次WRITE：三个种子复现</h1><p>仅复现已固定的soft教师主分支：当前context11教师概率驱动CBraMod的9600个LN参数，学习率0.01、梯度范数上限1，每个脉冲恢复原模型，评价未来四窗。源控制4人和源验证4人均已观察。新增29/43种子1024次WRITE，加原17种子共1536次；不开放在线或目标集更新。</p><p>先在每个人内平均三个种子的指标，再跨人汇总；不是概率集成，也不把种子当独立受试者。修正/伤害为种子平均的窗口出现次数，窗口可能重叠，不是整夜Macro-F1。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>按种子结果</h2>'+per_seed.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>固定判断</h2><pre>'+json.dumps(decision,indent=2)+'</pre><p>直接教师和固定半混合均使用相应未来窗口到达时的教师预测；这些预测不进入当前WRITE。教师有额外历史、架构、训练预算和计算成本。结果不能归因于TTT独有优势。</p><p><a href="protocol.json">协议</a> · <a href="summary_audit.json">统计审计</a></p>')
print(summary.to_string(index=False));print(per_seed[per_seed.variant=="teacher_soft_clip"].to_string(index=False));print(decision)

from common import *
import pandas as pd,hashlib
O=ROOT/"phase126";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip();audit=json.loads((O/"audit.json").read_text());assert audit["status"]=="PASS"
weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);rows=[];errors=[]
for a in audit["audits"]:
 path=O/f"seed{a['seed']}"/f"s{a['subject']:02d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==a["sha256"];d=np.load(path);assert d["write"].shape==(2,2,64,4,5)
 for ti,t in enumerate(d["times"]):
  y=d["labels"][ti,1:];assert (y>=0).all();b=d["base"][ti];tp=d["teacher"][ti,1:];candidates={"keep":b,"direct_teacher":tp,"static_half_blend":.5*b+.5*tp}
  for mi,method in enumerate(pr["methods"]):
   for li,lr in enumerate(pr["lrs"]):candidates[f"{method}_lr{lr:g}"]=d["write"][mi,li,ti]
  for variant,q in candidates.items():
   assert (q>0).all() and np.allclose(q.sum(1),1,atol=1e-12,rtol=0);delta=np.log(q[np.arange(4),y])-np.log(b[np.arange(4),y]);alt=-np.log(b[np.arange(4),y]).mean()+np.log(q[np.arange(4),y]).mean();errors.append(abs(delta.mean()-alt));bc=b.argmax(1)==y;qc=q.argmax(1)==y
   rows.append(dict(seed=a["seed"],group=a["group"],subject=a["subject"],t=int(t),variant=variant,future_ce_gain=float(delta.mean()),balanced_ce_gain=float((delta*weights[y]).mean()),accuracy=100*qc.mean(),corrected=int((~bc&qc).sum()),harmed=int((bc&~qc).sum())))
df=pd.DataFrame(rows);assert len(df)==10752;df.to_csv(O/"pulses.csv.gz",index=False,compression="gzip")
ps=df.groupby(["seed","group","subject","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index();ps.to_csv(O/"people_seeds.csv",index=False)
person=ps.groupby(["group","subject","variant"])[["future_ce_gain","balanced_ce_gain","accuracy","corrected","harmed"]].mean().reset_index();person.to_csv(O/"people.csv",index=False)
def summarize(data,keys):return data.groupby(keys).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index()
summary=summarize(person,["group","variant"]);summary.to_csv(O/"summary.csv",index=False);seedsummary=summarize(ps,["seed","group","variant"]);seedsummary.to_csv(O/"seed_summary.csv",index=False)
primary=f'{pr["primary"]["method"]}_lr{pr["primary"]["lr"]:g}';gates=[]
for group in ["controllers","validation"]:
 tab=summary[summary.group==group].set_index("variant");r=tab.loc[primary];useful=bool(r.future_ce_gain>0 and r.positive_people>=3 and r.balanced_ce_gain>=0 and r.corrected>=r.harmed and r.corrected>0);competitive=bool(all(r.future_ce_gain>=tab.loc[v,"future_ce_gain"] and r.accuracy>=tab.loc[v,"accuracy"] for v in ["direct_teacher","static_half_blend"]));gates.append(dict(group=group,useful_fresh_write=useful,competitive_with_static=competitive))
decision=dict(primary=primary,groups=gates,useful_all_groups=all(g["useful_fresh_write"] for g in gates),competitive_all_groups=all(g["competitive_with_static"] for g in gates),advance_online=False,advance_targets=False,reason="This fixed source experiment does not authorize online or target updates; largerLR/control cannot replace primary.")
(O/"decision.json").write_text(json.dumps(decision,indent=2));sa=dict(status="PASS",writes=6144,with_static_rows=len(df),independent_ce_algebra_error=max(errors),zero_write_bitwise_exact=all(a["zero_write_exact"] for a in audit["audits"]),max_static_head_compatibility_error=max(a["max_static_head_compatibility_error"] for a in audit["audits"]),perperson_seed_metric_averaging=True,no_online_macro_f1_claim=True)
(O/"summary_audit.json").write_text(json.dumps(sa,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1350px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>静态历史分类器上的单次熵WRITE</h1><p>三种子CBraMod配源选定历史残差头，在原8位已观察源受试者、各64脉冲上单次更新9600个LN参数，共6144次WRITE。当前整体预测先输出，再更新；固定主设置为历史分类器熵、学习率0.001，基础模型当前熵是目标函数对照，0.01为步长敏感性检查。不能从结果选取更有利设置替换主分支。</p><p>历史头固定。更新前的历史logit及已输出当前logit原样保留，后续每窗到达时才追加新参数产生的logit；不重算过去。启动边缘填充在当前梯度中正确重复当前向量。每个脉冲恢复原模型，未来四窗之间不再更新。零WRITE逐元素重放以及未来输入扰动的当前预测不变均通过。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>主分支逐种子</h2>'+seedsummary[seedsummary.variant==primary].to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<h2>固定判断</h2><pre>'+json.dumps(decision,indent=2)+'</pre><p>指标先平均脉冲、再在人内平均种子、最后平均受试者；修正/伤害为种子平均的未来窗出现次数。不是连续在线Macro-F1。直接光谱教师及固定半混合为强静态对照，教师未来预测只用于相应窗口的评分，未进入当前损失。历史头有额外源监督训练预算。</p><p><a href="protocol.json">固定协议及缓存语义</a> · <a href="summary_audit.json">统计和数值控制</a> · <a href="seed_summary.csv">所有逐种子结果</a></p>')
print(summary.to_string(index=False));print(seedsummary[seedsummary.variant==primary].to_string(index=False));print(decision);print(sa)

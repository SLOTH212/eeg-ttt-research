from common import *
import pandas as pd,hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O=ROOT/"phase115";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip()
audit=json.loads((O/"audit.json").read_text());assert audit["status"]=="PASS"
weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"])
ranges=[("next1",1,1),("next4",1,4),("next10",1,10),("next20",1,20),("next60",1,60),("bin1_4",1,4),("bin5_10",5,10),("bin11_20",11,20),("bin21_60",21,60)]
rows=[]; composition=[];max_error=0.
for seed in [17,29,43]:
 for group,sids in [("controllers",pr["controllers"]),("validation",pr["validation"])]:
  for sid in sids:
   path=O/f"seed{seed}"/f"s{sid:02d}.npz";d=np.load(path);a=json.loads(path.with_name(path.stem+"_audit.json").read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==a["artifact_sha256"]
   for i,t in enumerate(d["times"]):
    y=d["labels"][i];b=d["base"][i];w=d["write"][i];tp=d["teacher"][i];assert y[0]>=0
    candidates=dict(keep=b,write=w,teacher=tp,half_blend=.5*b+.5*tp,current_teacher_held=np.repeat(tp[:1],61,axis=0))
    for q in candidates.values():assert (q>0).all() and np.allclose(q.sum(1),1,atol=1e-12,rtol=0)
    for h,start,end in ranges:
     idx=np.arange(start,end+1);valid=idx[y[idx]>=0]
     composition.append(dict(seed=seed,group=group,subject=sid,t=int(t),horizon=h,scored=len(valid),available=len(idx),changed_stage=int((y[valid]!=y[0]).sum())))
     for stratum in ["all","same_current_stage","different_current_stage"]:
      jj=valid if stratum=="all" else valid[(y[valid]==y[0]) if stratum=="same_current_stage" else (y[valid]!=y[0])]
      if not len(jj):continue
      yy=y[jj];bce=-np.log(b[jj,yy]);bc=b[jj].argmax(1)==yy
      for variant,q in candidates.items():
       ce=-np.log(q[jj,yy]);delta=bce-ce;qc=q[jj].argmax(1)==yy
       rows.append(dict(seed=seed,group=group,subject=sid,t=int(t),horizon=h,start=start,end=end,stratum=stratum,variant=variant,n=len(jj),ce=float(ce.mean()),future_ce_gain=float(delta.mean()),balanced_ce_gain=float((delta*weights[yy]).mean()),accuracy=100*qc.mean(),accuracy_gain=100*(qc.astype(float)-bc).mean(),corrected=int((~bc&qc).sum()),harmed=int((bc&~qc).sum())))
df=pd.DataFrame(rows);df.to_csv(O/"pulses.csv.gz",index=False,compression="gzip")
keys=["seed","group","subject","horizon","start","end","stratum","variant"]
ps=df.groupby(keys).agg(ce=("ce","mean"),future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),accuracy_gain=("accuracy_gain","mean"),pulses=("t","size"),window_occurrences=("n","sum"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index()
ps.to_csv(O/"people_seeds.csv",index=False)
metric=["ce","future_ce_gain","balanced_ce_gain","accuracy","accuracy_gain","pulses","window_occurrences","corrected","harmed"]
person=ps.groupby(keys[1:])[metric].mean().reset_index();person.to_csv(O/"people.csv",index=False)
summary=person.groupby(["group","horizon","start","end","stratum","variant"]).agg(ce=("ce","mean"),future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy=("accuracy","mean"),accuracy_gain=("accuracy_gain","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())),people=("subject","size"),pulses=("pulses","sum"),window_occurrences=("window_occurrences","sum"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index()
summary.to_csv(O/"summary.csv",index=False)
comp=pd.DataFrame(composition);comp.to_csv(O/"composition.csv",index=False)
main=summary[(summary.stratum=="all")&summary.horizon.str.startswith("next")].sort_values(["group","end","variant"])
display=main[["group","horizon","variant","future_ce_gain","balanced_ce_gain","accuracy","accuracy_gain","positive_people"]]
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for axes,group in zip(ax,["controllers","validation"]):
 for variant,label in [("write","Single WRITE"),("teacher","Teacher on arrival"),("half_blend","Static half blend"),("current_teacher_held","Current teacher held")]:
  ss=main[(main.group==group)&(main.variant==variant)].sort_values("end")
  axes.plot(ss.end*.5,ss.accuracy_gain,marker="o",label=label)
 axes.axhline(0,color="gray",linewidth=.7);axes.set(xlabel="Cumulative future horizon (minutes)",ylabel="Accuracy change vs frozen (pp)",title=group.capitalize());axes.grid(alpha=.2)
ax[1].legend(fontsize=8);fig.suptitle("One isolated update; three-seed metrics averaged within 4 people/group")
fig.savefig(O/"horizon_effect.png",dpi=160);plt.close(fig)
strata=summary[(summary.variant=="write")&summary.horizon.isin(["bin1_4","bin5_10","bin11_20","bin21_60"])][["group","horizon","stratum","future_ce_gain","balanced_ce_gain","accuracy_gain","people","window_occurrences"]].sort_values(["group","horizon","stratum"])
exclusions={k:v["excluded_insufficient_duration"] for k,v in pr["eligibility"].items()}
summaryaudit=dict(status="PASS",pulses=audit["pulses"],max_legacy_probability_error=max(a["max_legacy_probability_error"] for a in audit["audits"]),max_batch_probability_error=max(a["max_batch_probability_error"] for a in audit["audits"]),batch_argmax_differences=sum(a["batch_argmax_differences"] for a in audit["audits"]),aggregation="Equal pulse metric means within person/seed then mean seeds within person then equal persons. Conditional strata omit empty pulse strata; not additive.",no_significance_search=True,no_online_or_target_rollout=True,excluded_duration_only=exclusions)
(O/"summary_audit.json").write_text(json.dumps(summaryaudit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}img{max-width:100%}</style><h1>单次教师WRITE的持续性：源数据固定观察范围</h1><p>三种子、源控制4人和源验证4人，均为此前已观察数据。仅保留原64个脉冲中其后有完整60窗的时点，按记录长度选择，与标签无关；所有观察范围使用同一批脉冲。单次当前soft教师WRITE后冻结参数至未来60窗，每个脉冲重新恢复原模型，不是连续在线更新。</p><p>1窗30秒，累计1/4/10/20/60窗以及不重叠时间段均事先固定。先算脉冲均值，再算人内均值、种子均值、组内人均值；未来窗可能重叠，不能视为独立样本。未评分标签仅从评分中排除。直接教师和半混合使用各窗口到达时的当前及历史信息；额外“保持当前教师概率”对照在整段使用更新时的同一分布。</p><img src="horizon_effect.png"><h2>累计范围</h2>'+display.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>WRITE分时段与阶段分层</h2>'+strata.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p>同/不同阶段仅比较未来真实阶段与更新时阶段，不表示中间从未发生转换，也不是实际可用门控。空分层的脉冲省略，因此条件指标不相加为总体。没有依据结果选择学习率、持续时长或重置规则；不重新开放已失败的在线/目标集推进门槛。</p><p>梯度使用原5窗批次以保持历史实验兼容；扩展61窗前向与5窗前向仅有浮点批量差异，前5窗保存原批次结果。兼容差值与分类变化见审计。</p><p><a href="protocol.json">固定协议</a> · <a href="summary_audit.json">审计</a> · <a href="summary.csv">完整统计</a></p>')
print(display[display.variant=="write"].to_string(index=False));print(strata.to_string(index=False));print(summaryaudit)

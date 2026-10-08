from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
from numpy.lib.stride_tricks import sliding_window_view
O=ROOT/"phase124";O.mkdir(exist_ok=True);cases=json.loads((ROOT/"phase85/protocol.json").read_text())["cases"];selected=[c for c in cases if c["montage"] in ["native","left"]];groups={"training":[0,1,2,10,11,20,21,30,31,32],"validation":[5,15,25,35],"controllers":[7,17,27,37]};records=[]
for group,sids in groups.items():
 for sid in sids:records.append(dict(group=group,subject=f"s{sid:02d}",labels=str(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_meta.npz")))
for c in selected:
 if c["domain"]!="source":records.append(dict(group=c["domain"],subject=c["subject"],labels=c["labels"]))
pr=dict(purpose="Retrospective temporaldistribution and accuracy-stratum diagnosis; no newmodel/inputdecision.",people=records,stage_names=["W","N1","N2","N3","REM"],statistics="Perperson scoredclassfractions, adjacentknown-label changefraction, stable11window fraction among fullyknown11windows,5x5conditional transitions. Complete observedstage runs exclude record-boundary/unknown-adjacent censoring; observed run lengths also retained.",performance="Reusephase120 CBr probabilities: history11 vsrepeat11 andfrozen, EMA1/.5, rawsource/SleepEDF and raw/full externalleft. Four fixedstrata: startupfirst10; knowncurrent withunscoredpast; fullyknownsame11stage; otherwise transition. Accuracydeltas exactlyrecompose bycount withinperson/seed.",aggregation="Metric meanswithinperson across3seeds thenequalpeople. Oneleftmontage perISRUCperson. Conditionalstrata withzero observations omitted, not interchangeable samples.",scope="Allpeoplelabels alreadyobserved. Source/targetdifferences descriptive, notcausalproof or deployabletrue-labelgate. No labelcrop, no resets or methodtuning.",cpu_ids=list(range(8,24)),gpu=False)
p=O/"protocol.json"
if p.exists():assert json.loads(p.read_text())==pr
else:p.write_text(json.dumps(pr,indent=2))
def labels(path):
 z=np.load(path);return z["y"] if str(path).endswith(".npz") else z
people=[];stage_rows=[];transition_rows=[];run_rows=[]
for rec in records:
 y=labels(rec["labels"]);known=y>=0;counts=np.bincount(y[known],minlength=5);adj=known[:-1]&known[1:];cm=np.bincount(y[:-1][adj]*5+y[1:][adj],minlength=25).reshape(5,5)
 windows=sliding_window_view(y,11);valid=(windows>=0).all(1);stable=(windows==windows[:,-1:]).all(1)&valid
 people.append(dict(group=rec["group"],subject=rec["subject"],windows=len(y),scored=int(known.sum()),wake_fraction=counts[0]/counts.sum(),adjacent_change_fraction=float((y[:-1][adj]!=y[1:][adj]).mean()),stable11_fraction=float(stable.sum()/valid.sum()),unknown_windows=int((~known).sum())))
 boundaries=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
 for start,end in zip(boundaries[:-1],boundaries[1:]):
  k=int(y[start])
  if k<0:continue
  censored=bool(start==0 or end==len(y) or y[start-1]<0 or y[end]<0)
  run_rows.append(dict(group=rec["group"],subject=rec["subject"],stage=pr["stage_names"][k],windows=int(end-start),minutes=.5*(end-start),censored=censored))
 for k,stage in enumerate(pr["stage_names"]):
  stage_rows.append(dict(group=rec["group"],subject=rec["subject"],stage=stage,count=int(counts[k]),fraction=float(counts[k]/counts.sum())))
  for j,dest in enumerate(pr["stage_names"]):transition_rows.append(dict(group=rec["group"],subject=rec["subject"],from_stage=stage,to_stage=dest,count=int(cm[k,j]),conditional_probability=float(cm[k,j]/cm[k].sum()) if cm[k].sum() else np.nan))
pd.DataFrame(people).to_csv(O/"people_temporal.csv",index=False);pd.DataFrame(stage_rows).to_csv(O/"people_stages.csv",index=False);pd.DataFrame(transition_rows).to_csv(O/"people_transitions.csv",index=False);runs=pd.DataFrame(run_rows);runs.to_csv(O/"runs.csv.gz",index=False,compression="gzip")
runperson=runs[~runs.censored].groupby(["group","subject","stage"]).agg(complete_runs=("windows","size"),median_minutes=("minutes","median"),mean_minutes=("minutes","mean")).reset_index();runperson.to_csv(O/"people_complete_runs.csv",index=False);runperson.groupby(["group","stage"]).agg(people=("subject","size"),complete_runs=("complete_runs","sum"),mean_person_median_minutes=("median_minutes","mean"),mean_person_mean_minutes=("mean_minutes","mean")).to_csv(O/"run_summary.csv")
temporal=pd.DataFrame(people).groupby("group").agg(people=("subject","size"),mean_scored_windows=("scored","mean"),wake_fraction=("wake_fraction","mean"),adjacent_change_fraction=("adjacent_change_fraction","mean"),stable11_fraction=("stable11_fraction","mean"));temporal.to_csv(O/"temporal_summary.csv")
stage_summary=pd.DataFrame(stage_rows).groupby(["group","stage"]).agg(people=("subject","size"),mean_fraction=("fraction","mean"),window_count=("count","sum")).reset_index();stage_summary.to_csv(O/"stage_summary.csv",index=False)
pd.DataFrame(transition_rows).groupby(["group","from_stage","to_stage"]).agg(people=("conditional_probability","count"),mean_probability=("conditional_probability","mean"),observed_count=("count","sum")).to_csv(O/"transition_summary.csv")
effects=[];recompose=[]
for c in selected:
 y=labels(c["labels"]);strata=np.full(len(y),"startup",dtype=object)
 for t in range(10,len(y)):
  h=y[t-10:t+1];strata[t]="unknown_history" if (h<0).any() else "stable" if np.all(h==h[-1]) else "transition"
 modes=["raw"] if c["domain"] in ["source","sleepedf"] else ["raw","full"];name="_".join(c[k] for k in ["domain","subject","montage"])
 for mode in modes:
  z=np.load(ROOT/"phase120/predictions"/f"{name}_{mode}.npz")
  for seed in [17,29,43]:
   for alpha in [1.,.5]:
    new=z[f"history11_seed{seed}"];new=new if alpha==1 else ema(new,alpha);nc=new.argmax(1)==y
    for ref in ["repeat11","frozen"]:
     old=z[f"{ref}_seed{seed}"];old=old if alpha==1 else ema(old,alpha);oc=old.argmax(1)==y;valid=y>=0;whole=100*np.mean(nc[valid].astype(float)-oc[valid]);weighted=0.
     for st in ["startup","unknown_history","stable","transition"]:
      mask=(strata==st)&valid;n=int(mask.sum())
      if not n:continue
      gain=100*np.mean(nc[mask].astype(float)-oc[mask]);weighted+=gain*n
      effects.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,seed=seed,ema=alpha,reference=ref,stratum=st,n=n,accuracy_gain=gain,corrected=int((nc[mask]&~oc[mask]).sum()),harmed=int((~nc[mask]&oc[mask]).sum())))
     err=abs(weighted/valid.sum()-whole);assert err<1e-12;recompose.append(err)
ef=pd.DataFrame(effects);ef.to_csv(O/"stratum_people_seeds.csv",index=False);pk=["domain","subject","montage","mode","ema","reference","stratum"];ep=ef.groupby(pk)[["n","accuracy_gain","corrected","harmed"]].mean().reset_index();ep.to_csv(O/"stratum_people.csv",index=False);es=ep.groupby(["domain","montage","mode","ema","reference","stratum"]).agg(people=("subject","size"),window_occurrences=("n","sum"),accuracy_gain=("accuracy_gain","mean"),corrected=("corrected","sum"),harmed=("harmed","sum")).reset_index();es.to_csv(O/"stratum_summary.csv",index=False)
audit=dict(status="PASS",people=len(records),whole_accuracy_weighted_recovery_error=max(recompose),classification_predictions_unchanged=True,no_label_based_gate=True,run_censoring_recorded=True)
(O/"audit.json").write_text(json.dumps(audit,indent=2));show=es[(es.ema==.5)&(es.reference=="repeat11")&es.stratum.isin(["stable","transition"])]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1350px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>队列时间结构与历史头效果的诊断</h1><p>源训练10人、源验证4人、控制4人、既有SleepEDF30人、健康10人、患者19人，共77人。均为此前已观察标签。ISRUC只用左侧，避免同一个人两侧重复计数。完整时间线保留；未知标签打断阶段连续段，不跨越未知标签计算相邻转换。</p><h2>人均时间结构</h2>'+temporal.to_html(float_format=lambda v:f"{v:.6f}")+'<h2>阶段比例</h2>'+stage_summary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>CBraMod历史头相对同容量重复头，EMA0.5</h2>'+show.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>稳定表示当前及前10窗真实阶段完全相同；转换表示这11窗均已评分但至少有阶段不同。启动和历史未知片段单列于CSV。分层只用于解释，真实阶段没有进入模型或动作门控；每个种子/人的准确率差按窗口数加权可严格恢复整体。分层Macro-F1不具有该加性，因此这里报告准确率差与修正/伤害。</p><p>阶段比例、转换频率和完整阶段段长均按人统计；完整段长排除记录端点或未知标签相邻的截断段。队列间同时存在输入和人群差异，这些描述不能证明时间结构单独导致跨库失败。<a href="run_summary.csv">完整段长</a> · <a href="transition_summary.csv">阶段转换矩阵</a> · <a href="audit.json">重组审计</a></p>')
print(temporal.to_string());print(show.to_string(index=False));print(audit)

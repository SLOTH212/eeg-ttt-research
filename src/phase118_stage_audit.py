from common import *
import pandas as pd,hashlib
O=ROOT/"phase118";O.mkdir(exist_ok=True);src=ROOT/"phase115";pr0=json.loads((src/"protocol.json").read_text());assert json.loads((src/"audit.json").read_text())["status"]=="PASS"
pr=dict(purpose="Retrospective stage-resolved classification audit of existing singleWRITE probabilities; no new method/inference or actiongate.",groups={"controllers":pr0["controllers"],"validation":pr0["validation"]},horizons=[4,60],methods=["keep","write","teacher","half_blend"],labels=["W","N1","N2","N3","REM"],aggregation="Pool scored future-window occurrences within eachperson/seed/horizon, form5x5confusion. Average seedmetricswithinperson, thenequalpeople; absentstage recall NaN, classF1 denominatorzero=>0, fixed5classMacroF1. This differs from equalpulsemean whenunscoredcounts vary.",scope="Previouslyobservedsourceonly. Overlappingfuturewindows are hypothetical outcomes of differentcoldupdates, not independent onlinepredictions. No onlineMacroF1claim." )
p=O/"protocol.json"
if p.exists():assert json.loads(p.read_text())==pr
else:p.write_text(json.dumps(pr,indent=2))
rows=[];stages=[];coverage=[];matrices=[]
for seed in [17,29,43]:
 for group,sids in pr["groups"].items():
  for sid in sids:
   d=np.load(src/f"seed{seed}"/f"s{sid:02d}.npz");folder=ROOT/("data/processed" if sid<20 else "phase2/processed");full_y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];fullcount=np.bincount(full_y[full_y>=0],minlength=5)
   for h in pr["horizons"]:
    yy=d["labels"][:,1:h+1].reshape(-1);good=yy>=0;y=yy[good];indexes=(d["times"][:,None]+np.arange(1,h+1)).reshape(-1)[good];count=np.bincount(y,minlength=5)
    if seed==17:
     for k,name in enumerate(pr["labels"]):coverage.append(dict(group=group,subject=sid,horizon=h,stage=name,pulse_window_occurrences=int(count[k]),fullnight_windows=int(fullcount[k]),pulse_fraction=float(count[k]/count.sum()),fullnight_fraction=float(fullcount[k]/fullcount.sum()),unique_future_windows=int(len(np.unique(indexes)))))
    base=d["base"][:,1:h+1].reshape(-1,5)[good];write=d["write"][:,1:h+1].reshape(-1,5)[good];teacher=d["teacher"][:,1:h+1].reshape(-1,5)[good]
    for variant,q in dict(keep=base,write=write,teacher=teacher,half_blend=.5*base+.5*teacher).items():
     pred=q.argmax(1);cm=np.bincount(y*5+pred,minlength=25).reshape(5,5);support=cm.sum(1);predcount=cm.sum(0);tp=np.diag(cm);den=support+predcount;f1=np.divide(2*tp,den,out=np.zeros(5),where=den>0);recall=np.divide(tp,support,out=np.full(5,np.nan),where=support>0)
     rows.append(dict(seed=seed,group=group,subject=sid,horizon=h,variant=variant,macro_f1=100*f1.mean(),balanced_accuracy=100*np.nanmean(recall),accuracy=100*tp.sum()/cm.sum(),ce=float(-np.log(q[np.arange(len(y)),y]).mean()),scored_occurrences=len(y),unique_future_windows=len(np.unique(indexes))))
     matrices.append(dict(seed=seed,group=group,subject=sid,horizon=h,variant=variant,confusion=cm.tolist()))
     for k,name in enumerate(pr["labels"]):
      kk=y==k;bc=base[kk].argmax(1)==k;qc=pred[kk]==k
      stages.append(dict(seed=seed,group=group,subject=sid,horizon=h,variant=variant,stage=name,support=int(support[k]),recall=100*recall[k],f1=100*f1[k],corrected=int((~bc&qc).sum()),harmed=int((bc&~qc).sum()),ce_gain=float((np.log(q[kk,k])-np.log(base[kk,k])).mean()) if kk.any() else np.nan))
df=pd.DataFrame(rows);df.to_csv(O/"people_seeds.csv",index=False);person=df.groupby(["group","subject","horizon","variant"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();person.to_csv(O/"people.csv",index=False);summary=person.groupby(["group","horizon","variant"])[["macro_f1","balanced_accuracy","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
st=pd.DataFrame(stages);st.to_csv(O/"stage_people_seeds.csv",index=False);sp=st.groupby(["group","subject","horizon","variant","stage"])[["support","recall","f1","corrected","harmed","ce_gain"]].mean().reset_index();sp.to_csv(O/"stage_people.csv",index=False);stage=sp.groupby(["group","horizon","variant","stage"]).agg(support=("support","sum"),recall=("recall","mean"),f1=("f1","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),ce_gain=("ce_gain","mean"),people_with_stage=("recall","count")).reset_index();stage.to_csv(O/"stage_summary.csv",index=False);pd.DataFrame(coverage).to_csv(O/"coverage.csv",index=False);(O/"confusions.json").write_text(json.dumps(matrices))
deltas=[]
for group in pr["groups"]:
 for h in pr["horizons"]:
  sub=person[(person.group==group)&(person.horizon==h)];table=sub.pivot(index="subject",columns="variant",values="macro_f1")
  for ref in ["keep","teacher","half_blend"]:
   delta=table.write-table[ref];deltas.append(dict(group=group,horizon=h,reference=ref,macro_f1_delta=float(delta.mean()),positive_people=int((delta>0).sum()),worst=float(delta.min())))
delta=pd.DataFrame(deltas);delta.to_csv(O/"macro_f1_deltas.csv",index=False)
audit=dict(status="PASS",source_only=True,new_inference=False,confusion_rows=len(matrices),current_future_label_not_used_for_method=True,independent_events_claim=False)
(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>单次WRITE的阶段分类审计</h1><p>复用固定phase115预测，不新增训练或推理。对每个人、种子、观察范围，将有效未来窗出现次数构成五类混淆矩阵，再先平均种子、后平均人。这里是独立假想更新的结果，不是连续在线Macro-F1；60窗之间有重复时刻。与逐脉冲均值相比，本表在未评分窗存在时权重略有差异。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>WRITE与各对照的Macro-F1差</h2>'+delta.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>WRITE逐阶段</h2>'+stage[stage.variant=="write"].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>支持数、修正/伤害为种子平均的出现次数；某人没有某阶段时，该阶段recall为缺失，不虚构样本；宏F1固定五类、无分母项记0。阶段标签仅用于回顾评分。<a href="coverage.csv">脉冲与整夜阶段组成</a> · <a href="confusions.json">混淆矩阵</a> · <a href="protocol.json">协议</a></p>')
print(delta.to_string(index=False));print(stage[(stage.variant=="write")&(stage.horizon==4)].to_string(index=False))

from common import *
from phase2_temporal import ema
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase70";pr=json.loads((o/"protocol.json").read_text());assert (o/"complete.json").exists();assert hashlib.sha256((ROOT/"phase66/selection.json").read_bytes()).hexdigest()==pr["selection_file_sha256"]
for choice in pr["selection"].values():assert hashlib.sha256((Path(choice["folder"])/"model.pt").read_bytes()).hexdigest()==choice["checkpoint_sha256"]
rows=[]
for sid in pr["subjects"]:
 z=np.load(o/f"s{sid:02d}.npz");meta=json.loads((o/f"s{sid:02d}.json").read_text());ref=np.load(ROOT/"phase40"/f"s{sid:02d}.npz");root=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"]
 assert np.array_equal(y,z["y"]) and np.array_equal(y,ref["y"]) and meta["state_unchanged"] and meta["no_online_updates"]
 for model,basep in [("pretrained",z["pretrained"]),("random",z["random"]),("labram",ref["frozen"])]:
  assert np.isfinite(basep).all() and abs(basep.sum(1)-1).max()<1e-5
  for smooth in pr["ema_values"]:
   p=basep if smooth==1 else ema(basep,smooth);m=metrics(y,p);valid=y>=0
   if model!="labram":assert abs(m["macro_f1"]-meta["methods"][model+f"_ema{smooth:g}"]["macro_f1"])<1e-12
   rows.append(dict(subject=sid,cohort=meta["cohort"],model=model,ema=smooth,macro_f1=100*m["macro_f1"],balanced_accuracy=100*m["balanced_accuracy"],accuracy=100*float((p[valid].argmax(1)==y[valid]).mean()),cross_entropy=m["ce"],**{f"recall_{i}":100*v for i,v in enumerate(m["recall"])}))
df=pd.DataFrame(rows);assert len(df)==270;df.to_csv(o/"participants.csv",index=False);means=df.groupby(["model","ema"])[["macro_f1","balanced_accuracy","accuracy","cross_entropy"]].mean();means.to_csv(o/"summary.csv")
cohorts=df.groupby(["cohort","model","ema"])[["macro_f1","balanced_accuracy","accuracy","cross_entropy"]].mean();cohorts.to_csv(o/"cohort_summary.csv")
def contrast(a,b,cohort,comparison):
 assert not a.isna().any() and not b.isna().any();delta=(a-b).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta))).mean(1),[.025,.975]);return dict(cohort=cohort,comparison=comparison,n_people=len(delta),method_f1=float(a.mean()),reference_f1=float(b.mean()),difference_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_person_pp=float(delta.min()))
comparisons=[]
for cohort in ["all","young","middle","older"]:
 g=df if cohort=="all" else df[df.cohort==cohort]
 for smooth in [1.,.8,.5]:
  a=g[(g.model=="pretrained")&(g.ema==smooth)].set_index("subject").macro_f1.sort_index()
  for reference in ["labram","random"]:
   b=g[(g.model==reference)&(g.ema==smooth)].set_index("subject").macro_f1.reindex(a.index)
   comparisons.append(contrast(a,b,cohort,f"pretrained_vs_{reference}_ema{smooth:g}"))
comp=pd.DataFrame(comparisons);comp.to_csv(o/"paired_comparisons.csv",index=False)
# Explicit supplementary historical practical baseline, not a replacement primary.
old=pd.concat([pd.read_csv(ROOT/"phase4/all_metrics.csv"),pd.read_csv(ROOT/"phase6/all_metrics.csv")],ignore_index=True);historical=[]
for cohort in ["all","young","middle","older"]:
 g=df if cohort=="all" else df[df.cohort==cohort]
 for smooth,method in [(1.,"frozen"),(.8,"ema_08")]:
  a=g[(g.model=="pretrained")&(g.ema==smooth)].set_index("subject").macro_f1.sort_index();ref=old[(old.source=="mixed")&(old.variant=="context11")&(old.seed==17)&(old.method==method)].set_index("subject").f1
  assert not ref.index.duplicated().any();b=100*ref.reindex(a.index);historical.append(contrast(a,b,cohort,f"pretrained_vs_historical_context11_ema{smooth:g}"))
hist=pd.DataFrame(historical);hist.to_csv(o/"historical_context_comparisons.csv",index=False)
audit=dict(status="PASS",subjects=30,classifiers=2,probability_method_streams_recomputed=270,source_selection_and_checkpoint_hashes_verified=True,true_labels_and_model_state_checks=True,no_online_updates=True,historical_reference="previously audited phase4/6 mixedseed17 context11 metrics; supplementary, not same history capacity",bootstrap="subject resampling conditional on oneclassifierseed and prior exploratory selection; not uncertainty from all tuning")
(o/"audit.json").write_text(json.dumps(audit,indent=2))
primary=comp[(comp.cohort=="all")&(comp.comparison=="pretrained_vs_labram_ema0.5")].iloc[0].to_dict();(o/"primary_result.json").write_text(json.dumps(primary,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1350px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>固定CBraMod分类器的既有目标人群迁移</h1><p>来源十人训练、四人选择的检查点与EMA网格在目标评估前冻结。30名此前已观察的受试者，预训练及随机CBraMod均纯静态前向；标准熵新鲜写入失败，没有把该更新推广到目标人群。主比较为预训练CBraMod EMA.5与相同来源LaBraM EMA.5，其他固定EMA和随机载体作次级比较。</p>'+means.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>按人配对：全体</h2>'+comp[comp.cohort=="all"].to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<h2>年龄群体</h2>'+comp[comp.cohort!="all"].to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<h2>补充历史强时序基线</h2>'+hist.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<p>历史11窗频谱模型有约5分钟历史，两者都叠加同EMA时仍不等于相同历史表达能力；调参预算、模型与预训练数据也不同。此补充不替换预定主比较。按人bootstrap区间只描述当前固定方案，不涵盖单一训练种子和反复探索选择的不确定性；均值不能排除个体损害。</p><p>这是静态分类迁移，不是TTT或学习保留成功证据；目标人群已被项目多次观察，不是新盲测。结果不用于重选微调轮次、步长或EMA。</p><p><a href="protocol.json">目标评估前锁定</a> · <a href="primary_result.json">主比较</a> · <a href="audit.json">独立审计</a> · <a href="participants.csv">逐人结果</a></p></html>';(o/"report.html").write_text(html)
print(means.to_string());print(comp[comp.cohort=="all"].to_string(index=False));print(hist[hist.cohort=="all"].to_string(index=False));print(primary)

from common import *
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase33";source_rows=[];target_rows=[];audits=[];finished=0
protocol=json.loads((ROOT/"phase31/protocol.json").read_text())
for seed in [17,29,43]:
 for weight in [0.,.01]:
  folder=ROOT/("phase28" if seed==17 else "phase33")/(f"young_joint_{weight:g}" if seed==17 else f"young_joint_{weight:g}_seed{seed}");meta=json.loads((folder/"complete.json").read_text());validation=json.loads((folder/"validation_audit.json").read_text());assert validation["status"]=="PASS";assert hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()==validation["selected_checkpoint_sha256"];assert meta["best_val_f1"]==max([meta["initial_val_f1"]]+[h["val_f1"] for h in meta["history"]]);source_rows.append(dict(seed=seed,weight=weight,epoch=meta["best_epoch"],f1=100*meta["best_val_f1"],masked_ce=float(np.mean([v["masked_ce"] for v in meta["selected_auxiliary"].values()]))))
  evaluation=ROOT/"phase31"/f"young_joint_{weight:g}" if seed==17 else o/"evaluation"/f"young_joint_{weight:g}_seed{seed}"
  if not (evaluation/"complete.json").exists():continue
  finished+=1
  if seed!=17:
   lock=json.loads((evaluation/"selection.json").read_text());assert lock["training_seed"]==seed and lock["no_new_lr_search"];assert lock["checkpoints"][str(weight)]["sha256"]==validation["selected_checkpoint_sha256"]
  for sid in protocol["subjects"]:
   record=json.loads((evaluation/f"s{sid:02d}.json").read_text());saved=np.load(evaluation/f"s{sid:02d}.npz");y=saved["y"];root=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(y,np.load(root/f"s{sid:02d}_meta.npz")["y"]);assert record["nonwritable_unchanged"] and record["radius_verified"]
   for method in protocol["methods"]:
    p=saved[method];assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;mm=metrics(y,p);assert abs(mm["macro_f1"]-record["methods"][method]["macro_f1"])<1e-12;target_rows.append(dict(seed=seed,weight=weight,subject=sid,cohort=record["cohort"],method=method,f1=mm["macro_f1"],ce=mm["ce"]));audits.append(dict(seed=seed,weight=weight,subject=sid,method=method))
source=pd.DataFrame(source_rows);source.to_csv(o/"source_summary.csv",index=False);df=pd.DataFrame(target_rows);df.to_csv(o/"target_participants.csv",index=False)
(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=finished==6,completed_carriers=finished,method_streams=len(audits),source_checkpoint_choices_verified=True,checkpoint_hashes_verified=True,raw_labels_match=True,metrics_recomputed=True),indent=2))
header="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>联合训练固定设置复现</title><style>body{font:15px/1.65 system-ui;max-width:1300px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ccc}</style><h1>固定权重0/0.01：三个训练种子的复现</h1><p>年轻来源，相同预训练初始化、数据、编码器LR2e-5、10轮预算，只改变训练随机数17/29/43。每个模型仅根据源验证选epoch，不再选择辅助权重。后两种子的在线设置固定为seed17源选择：零权重不更新；联合模型熵1e-5、掩码1e-4。所有人群此前均已观察，属于探索性复现。</p><h2>源验证选择</h2>"""
if finished!=6:
 (o/"report.html").write_text(header+source.to_html(index=False,float_format=lambda x:f"{x:.4f}")+f"<p>当前{finished}/6个模型完成目标评估。三种子配对汇总等待全部完成。</p></html>");print("SOURCE",source.to_string(index=False));print("TARGET COMPLETE",finished,"/6");raise SystemExit(0)
seedmean=df.groupby(["seed","weight","cohort","method"])[["f1","ce"]].mean().reset_index();allmean=df.groupby(["seed","weight","method"])[["f1","ce"]].mean().reset_index();allmean["cohort"]="all";seedmean=pd.concat([seedmean,allmean],ignore_index=True);seedmean["f1"]*=100;seedmean.to_csv(o/"seed_means.csv",index=False)
assert (df.groupby(["weight","subject","cohort","method"]).seed.nunique()==3).all();mean=df.groupby(["weight","subject","cohort","method"])[["f1","ce"]].mean().reset_index();mean.to_csv(o/"mean3_person_metrics.csv",index=False);contrasts=[]
def record(delta,cohort,comparison,method_mean,reference_mean):
 rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);return dict(cohort=cohort,comparison=comparison,n_people=len(delta),method_f1=method_mean,reference_f1=reference_mean,delta_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=float(delta.min()))
for cohort in ["all","young","middle","older"]:
 g=mean if cohort=="all" else mean[mean.cohort==cohort]
 for method in ["frozen","weak_ema","strong_ema"]:
  q=g[g.method==method].pivot(index="subject",columns="weight",values="f1");delta=100*(q[.01]-q[0.]).to_numpy();contrasts.append(record(delta,cohort,"joint_vs_zero_"+method,100*q[.01].mean(),100*q[0.].mean()))
 q=g[g.weight==.01].pivot(index="subject",columns="method",values="f1")
 for method,ref,label in [("masked_selected","frozen","masked_vs_frozen"),("entropy_selected","frozen","entropy_vs_frozen"),("masked_selected_ema08","weak_ema","maskedEMA_vs_sameEMA"),("entropy_selected_ema08","weak_ema","entropyEMA_vs_sameEMA"),("masked_selected","strong_ema","masked_vs_fixed_strongEMA")]:
  delta=100*(q[method]-q[ref]).to_numpy();contrasts.append(record(delta,cohort,label,100*q[method].mean(),100*q[ref].mean()))
comp=pd.DataFrame(contrasts);comp.to_csv(o/"paired_comparisons.csv",index=False)
html=header+source.to_html(index=False,float_format=lambda x:f"{x:.4f}")+"""<h2>30人、先按个人平均三个种子的性能，再作配对</h2><p>这里平均的是种子各自的性能，不是三模型预测集成。bootstrap只按受试者重采样，不能完整表达种子总体和研究选择的不确定性。固定强EMA是沿用原源选择的alpha0.5，完整CSV另提供同平滑条件比较。</p>"""+comp[comp.cohort=="all"].to_html(index=False,float_format=lambda x:f"{x:.4f}")+"<h2>各种子的静态迁移分数</h2>"+seedmean[(seedmean.cohort=="all")&(seedmean.method=="frozen")].to_html(index=False,float_format=lambda x:f"{x:.4f}")+"""<p>六个模型、1260条方法预测流复算通过。两次早期失败的锁定主检验、单种子迁移结果均保留；此处增加训练随机性证据。</p><a href="paired_comparisons.csv">所有分组对照</a> · <a href="seed_means.csv">全部种子均值</a> · <a href="audit.json">审计</a></html>""";(o/"report.html").write_text(html);print(comp[comp.cohort=="all"].to_string(index=False));print(seedmean[(seedmean.cohort=="all")&(seedmean.method=="frozen")].to_string(index=False));print("COMPLETE 1260")

from common import *
import pandas as pd,hashlib,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase31";lock=json.loads((o/"selection_lock.json").read_text());protocol=json.loads((o/"protocol.json").read_text());rows=[];checks=[]
assert hashlib.sha256((ROOT/"phase30/source_selection.json").read_bytes()).hexdigest()==lock["source_selection_sha256"]
for weight in [0.,.01]:
 assert hashlib.sha256(Path(lock["checkpoints"][str(weight)]["path"]).read_bytes()).hexdigest()==lock["checkpoints"][str(weight)]["sha256"]
 for f in sorted((o/f"young_joint_{weight:g}").glob("s*.json")):
  d=json.loads(f.read_text());sid=d["subject"];assert sid in protocol["subjects"] and sid not in protocol["source_controllers_excluded"];saved=np.load(f.with_suffix(".npz"));y=saved["y"];root=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(y,np.load(root/f"s{sid:02d}_meta.npz")["y"]);assert d["nonwritable_unchanged"] and d["radius_verified"] and d["source_selection_locked"]
  for method in protocol["methods"]:
   p=saved[method];assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;mm=metrics(y,p);assert abs(mm["macro_f1"]-d["methods"][method]["macro_f1"])<1e-12;rows.append(dict(weight=weight,subject=sid,cohort=d["cohort"],method=method,f1=mm["macro_f1"],ce=mm["ce"],balanced_accuracy=mm["balanced_accuracy"]));checks.append(dict(weight=weight,subject=sid,method=method))
if not rows:raise SystemExit("No completed transfer people")
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False);summary=df.groupby(["weight","cohort","method"])[["f1","ce"]].mean().reset_index();summary["f1"]*=100;summary.to_csv(o/"summary.csv",index=False);contrasts=[]
def paired(g,method,ref,label,weight,cohort):
 q=g.pivot(index="subject",columns="method",values="f1")[[method,ref]].dropna();delta=100*(q[method]-q[ref]).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);return dict(weight=weight,cohort=cohort,comparison=label,n=len(q),method_f1=100*q[method].mean(),reference_f1=100*q[ref].mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],wins=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min())
for weight,g in df.groupby("weight"):
 choice=next(d for d in lock["source_selection"]["decisions"] if d["weight"]==weight);beststatic=max([("frozen",choice["frozen_f1"]),("weak_ema",choice["weak_ema_f1"]),("strong_ema",choice["strong_ema_f1"])],key=lambda z:z[1])[0]
 for cohort in ["all","young","middle","older"]:
  h=g if cohort=="all" else g[g.cohort==cohort]
  if h.empty:continue
  for method,ref,label in [("masked_selected",beststatic,"masked_vs_source_best_static"),("masked_selected","frozen","masked_vs_frozen"),("entropy_selected","frozen","entropy_vs_frozen"),("masked_selected_ema08","weak_ema","maskedEMA_vs_sameEMA"),("entropy_selected_ema08","weak_ema","entropyEMA_vs_sameEMA")]:contrasts.append(paired(h,method,ref,label,weight,cohort))
transfer=[]
for method in ["frozen","weak_ema","strong_ema"]:
 for cohort in ["all","young","middle","older"]:
  g=df[df.method==method];g=g if cohort=="all" else g[g.cohort==cohort];q=g.pivot(index="subject",columns="weight",values="f1")
  if not {0.,.01}<=set(q.columns):continue
  q=q[[0.,.01]].dropna()
  if q.empty:continue
  delta=100*(q[.01]-q[0.]).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);transfer.append(dict(cohort=cohort,method=method,n=len(q),joint_f1=100*q[.01].mean(),zero_f1=100*q[0.].mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1]))
comp=pd.DataFrame(contrasts);comp.to_csv(o/"paired_comparisons.csv",index=False);static=pd.DataFrame(transfer);static.to_csv(o/"static_transfer.csv",index=False);complete=len(checks)==420;(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=complete,saved_method_streams=len(checks),new_person_models=len(checks)//7,locked_source_selection_hash_verified=True,model_hashes_verified=True,labels_match_raw=True,metrics_recomputed=True,details=checks),indent=2))
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>修正辅助模型的探索性迁移</title><style>body{font:15px/1.65 system-ui;max-width:1300px;margin:30px auto}td,th{padding:6px;border-bottom:1px solid #ccc}</style><h1>30名既有非控制器受试者：锁定源选择后的迁移</h1><p>所有人群已在此前项目中观察过，这是探索性迁移，不是新盲测。两次早期锁定主检验的失败结论保持不变。源训练、辅助权重与在线学习率先完成源验证并保存哈希锁定，再读取这些人的预测数据。当前窗口先预测后更新；目标真实标签只用于离线评分。</p><p>预设主要探索比较为权重0.01模型的掩码适应，相对源验证选出的最佳静态对照（冻结/弱EMA/强EMA）；其余为次要诊断。全部按人配对bootstrap，仅反映当前人群差异，不覆盖反复研究选择及训练种子不确定性。</p>"""+f"<p>已完成{len(checks)//7}/60个人模型案例，{len(checks)}条方法预测流，审计通过。</p>"+comp[(comp.weight==.01)&comp.comparison.isin(["masked_vs_source_best_static","masked_vs_frozen","maskedEMA_vs_sameEMA"])].to_html(index=False,float_format=lambda x:f"{x:.4f}")+"<h2>静态联合训练相对匹配零权重</h2>"+static.to_html(index=False,float_format=lambda x:f"{x:.4f}")+"""<p><a href="selection_lock.json">源选择锁定</a> · <a href="paired_comparisons.csv">全部方法比较</a> · <a href="audit.json">审计</a></p></html>""";(o/"report.html").write_text(html);print(comp[(comp.weight==.01)&(comp.cohort=="all")].to_string(index=False));print(static[static.cohort=="all"].to_string(index=False));print("STREAMS",len(checks),"COMPLETE",complete)

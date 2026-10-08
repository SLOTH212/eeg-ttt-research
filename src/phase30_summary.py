from common import *
from phase2_temporal import ema
import pandas as pd,warnings
warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true")
o=ROOT/"phase30";rows=[];checks=[];config=[]
for f in sorted(o.glob("young_w*/complete.json")):
 d=json.loads(f.read_text());weight=d["weight"];objective=d["objective"];lr=d["lr"];scores=[]
 for record in d["people"]:
  sid=record["subject"];saved=np.load(f.parent/f"s{sid:02d}.npz");base=np.load(ROOT/"phase28"/f"young_joint_{weight:g}"/f"s{sid:02d}.npz");y=saved["y"];p=saved["p"];assert np.array_equal(y,base["y"]);assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;assert np.isfinite(saved["loss"]).all() and np.isfinite(saved["gradient_norm"]).all();m=metrics(y,p);assert abs(m["macro_f1"]-record["metrics"]["macro_f1"])<1e-12;assert record["nonwritable_unchanged"] and record["radius_verified"];scores.append(m["macro_f1"])
  for method,pr in [("adapted",p),("adapted_ema08",ema(p,.8)),("frozen",base["p"]),("weak_ema",ema(base["p"],.8)),("strong_ema",ema(base["p"],.5))]:
   mm=metrics(y,pr);rows.append(dict(weight=weight,objective=objective,lr=lr,subject=sid,method=method,f1=mm["macro_f1"],ce=mm["ce"],balanced_accuracy=mm["balanced_accuracy"]))
  checks.append(dict(weight=weight,objective=objective,lr=lr,subject=sid))
 assert abs(np.mean(scores)-d["mean_f1"])<1e-12;config.append(dict(weight=weight,objective=objective,lr=lr))
if not rows:raise SystemExit("No complete source trajectory configs")
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False);summary=df.groupby(["weight","objective","lr","method"])[["f1","ce","balanced_accuracy"]].mean().reset_index();summary["f1"]*=100;summary.to_csv(o/"summary.csv",index=False);decisions=[]
for (weight,objective),g in summary.groupby(["weight","objective"]):
 if len(g.lr.unique())!=4:continue
 baseline=g[g.method=="frozen"].f1.iloc[0];weak=g[g.method=="weak_ema"].f1.iloc[0];strong=g[g.method=="strong_ema"].f1.iloc[0];candidates=[dict(lr=0.,f1=float(baseline))]+g[g.method=="adapted"][["lr","f1"]].to_dict("records");selected=max(candidates,key=lambda x:x["f1"]);decisions.append(dict(weight=weight,objective=objective,selected_lr=selected["lr"],selected_f1=selected["f1"],frozen_f1=baseline,weak_ema_f1=weak,strong_ema_f1=strong,delta_vs_weak_pp=selected["f1"]-weak,candidates=candidates))
(o/"source_selection.json").write_text(json.dumps(dict(source_only=True,complete=len(config)==16,selection_rule="Mean source4person raw adapted F1 including freeze; EMA onadapted onlysecondary diagnostic",decisions=decisions),indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=len(config)==16,complete_configs=len(config),audited_adapted_streams=len(checks),checks="Labels, finite normalized probabilities, exact F1 recomputation, saved runtime frozen-state/radius checks",details=checks),indent=2))
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>联合模型的完整在线源验证</title><style>body{font:15px/1.65 system-ui;max-width:1200px;margin:30px auto}td,th{padding:6px;border-bottom:1px solid #ccc}</style><h1>完整记录：先预测、后更新</h1><p>年轻来源联合权重0与0.01，分别进行熵/修正教师掩码更新，4个学习率，4名源验证人员。每条记录从源参数开始，未知睡眠标签窗口仍更新；任务标签只用于离线评分。只写LayerNorm，其余编码器与两个头固定，梯度裁剪1，源半径0.1RMSfloor1。</p><p>每种模型/目标按源平均F1选择学习率，包含冻结回退。冻结、弱EMA和强EMA是同一模型的静态参考。对适应预测再加弱EMA是源域补充分析，不改原轨迹和主选择规则。所有源验证已反复使用；不是目标盲测。</p>"""+f"<p>完成{len(config)}/16组，{len(checks)}条完整预测流。</p>"+pd.DataFrame(decisions).drop(columns=["candidates"],errors="ignore").to_html(index=False,float_format=lambda x:f"{x:.4f}")+"<h2>全部源验证分数</h2>"+summary.pivot(index=["weight","objective","lr"],columns="method",values="f1").reset_index().to_html(index=False,float_format=lambda x:f"{x:.4f}")+"""<p><a href="audit.json">预测审计</a> · <a href="graph_audit.json">掩码图数值审计</a> · <a href="frozen_prefix_audit.json">冻结与前缀审计</a></p></html>""";(o/"report.html").write_text(html)
print(pd.DataFrame(decisions).drop(columns=["candidates"],errors="ignore").to_string(index=False));print("COMPLETE",len(config),"/16")

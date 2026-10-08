from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
o=ROOT/"phase37";protocol=json.loads((o/"protocol.json").read_text());rows=[];audits=[]
for folder in sorted(o.iterdir()):
 if not folder.is_dir() or not (folder/"complete.json").exists():continue
 meta=json.loads((folder/"complete.json").read_text());lock=json.loads((folder/"selection.json").read_text());source=meta["source"];variant=meta["variant"];lr=meta["lr"];checkpoint=ROOT/"phase35"/source/"model.pt" if variant=="fitted" else ROOT/"phase28"/f"{source}_joint_0"/"model.pt";assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==lock["checkpoint_sha256"]
 for sid in protocol["subjects"]:
  rec=json.loads((folder/f"s{sid:02d}.json").read_text());z=np.load(folder/f"s{sid:02d}.npz");base=np.load(ROOT/"phase28"/f"{source}_joint_0"/f"s{sid:02d}.npz");root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];assert np.array_equal(y,z["y"]) and np.array_equal(y,base["y"]);assert np.isfinite(z["p"]).all() and abs(z["p"].sum(1)-1).max()<1e-5;assert rec["nonwritable_unchanged"] and rec["radius_verified"]
  pp={"metrics":z["p"],"frozen":base["p"],"weak_ema":ema(base["p"],.8),"strong_ema":ema(base["p"],.5),"adapted_ema":ema(z["p"],.8)}
  vals={}
  for method,p in pp.items():
   mm=metrics(y,p);assert abs(mm["macro_f1"]-rec[method]["macro_f1"])<1e-12;vals[method]=mm["macro_f1"]*100
  row=dict(source=source,variant=variant,lr=lr,subject=sid,**vals,ce=metrics(y,z["p"])["ce"]);row["delta_frozen_pp"]=vals["metrics"]-vals["frozen"];row["delta_weak_ema_pp"]=vals["metrics"]-vals["weak_ema"];row["delta_matched_ema_pp"]=vals["adapted_ema"]-vals["weak_ema"];rows.append(row);audits.append(dict(case=folder.name,subject=sid))
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False);means=df.groupby(["source","variant","lr"]).mean(numeric_only=True).drop(columns="subject").reset_index();means.to_csv(o/"summary.csv",index=False);complete=len(audits)==48;decisions=[]
for source in protocol["sources"]:
 g=df[(df.source==source)&(df.variant=="fitted")]
 for lr in protocol["lrs"]:
  h=g[g.lr==lr]
  if len(h)!=4:continue
  mean=h["metrics"].mean();best_static=max(h[k].mean() for k in ["frozen","weak_ema","strong_ema"]);positive=int((h.delta_frozen_pp>0).sum());decisions.append(dict(source=source,lr=lr,mean_f1=mean,best_static=best_static,positive_people_vs_frozen=positive,advance=bool(mean>best_static and positive>=3)))
(o/"advance_decisions.json").write_text(json.dumps(dict(complete=complete,decisions=decisions,criterion=protocol["selection"]),indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=complete,adapted_streams=len(audits),recomputed_methods_per_stream=5,hashes_labels_metrics_verified=True,nonwritable_and_radius_assertions=True),indent=2))
style='<style>body{font:15px/1.6 system-ui;max-width:1300px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}section{overflow:auto}</style>'
html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>辅助头训练前后：完整连续在线源验证</h1><p>两来源×两辅助版本×三个固定步长0.0001/0.001/0.01，四名既有源验证受试者。每个当前窗口先预测，再用该窗口的固定教师目标更新编码器LayerNorm；分类器原始参数相同，辅助头固定。未知标签窗口也正常处理，标签只用于最后评分。</p><p>冻结、弱EMA0.8、强EMA0.5、相同步长的未训练辅助头均为对照。推进规则在运行前固定：拟合辅助头的在线分数高于全部静态对照，且至少3/4人优于自身冻结。四人反复用于源选择，结论仍属探索，不是新盲测。</p>'+f'<p>当前完成{len(audits)}/48条适应轨迹，每条复算5种方法。执行前4组64窗口的GPU图与普通执行预测、最终参数均完全一致。</p><section>'+means.to_html(index=False,float_format=lambda v:f"{v:.5f}")+'</section><h2>预设推进判定</h2><section>'+pd.DataFrame(decisions).to_html(index=False,float_format=lambda v:f"{v:.5f}")+'</section><p><a href="participants.csv">逐人结果</a> · <a href="audit.json">独立审计</a> · <a href="graph_audit.json">执行一致性</a> · <a href="protocol.json">协议</a></p></html>'
(o/"report.html").write_text(html);print(means.to_string(index=False));print("COMPLETE",complete,len(audits));print(decisions)

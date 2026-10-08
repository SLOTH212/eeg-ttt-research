from common import *
import pandas as pd,hashlib
o=ROOT/"phase40";protocol=json.loads((o/"protocol.json").read_text());lock=json.loads((o/"selection_lock.json").read_text());assert hashlib.sha256((ROOT/"phase39/source_selection.json").read_bytes()).hexdigest()==lock["source_selection_sha256"]
for ck in lock["checkpoints"].values():assert hashlib.sha256(Path(ck["path"]).read_bytes()).hexdigest()==ck["sha256"]
rows=[];finished=0
for sid in protocol["subjects"]:
 done=o/f"s{sid:02d}.json"
 if not done.exists():continue
 rec=json.loads(done.read_text());z=np.load(o/f"s{sid:02d}.npz");root=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(z["y"],np.load(root/f"s{sid:02d}_meta.npz")["y"]);assert rec["nonwritable_unchanged"] and rec["radius_verified"] and rec["source_selection_locked"] and rec["pre_update_predictions"]
 for method in protocol["methods"]:
  p=z[method];assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;mm=metrics(z["y"],p);assert abs(mm["macro_f1"]-rec["methods"][method]["macro_f1"])<1e-12;valid=z["y"]>=0;yp=z["y"][valid];bp=z["frozen"][valid].argmax(1);ap=p[valid].argmax(1);rows.append(dict(subject=sid,cohort=rec["cohort"],method=method,f1=100*mm["macro_f1"],ce=mm["ce"],accuracy=float((ap==yp).mean()),changed_fraction=float((bp!=ap).mean()),corrected_fraction=float(((bp!=yp)&(ap==yp)).mean()),harmed_fraction=float(((bp==yp)&(ap!=yp)).mean())))
 finished+=1
d=pd.DataFrame(rows);d.to_csv(o/"participants.csv",index=False);complete=finished==len(protocol["subjects"]);(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=complete,subjects=finished,method_streams=len(rows),checkpoint_and_selection_hashes_verified=True,raw_labels_verified=True,metrics_recomputed=True,nonwritable_and_radius_assertions=True),indent=2))
style='<style>body{font:15px/1.65 system-ui;max-width:1350px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ccc}section{overflow:auto}</style>'
html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>分类器保持不变、仅拟合辅助头：混合来源迁移</h1><p>源验证锁定：拟合辅助头掩码LR0.01、未拟合辅助头源选择LR0.001，熵更新选择不更新。额外报告未拟合头LR0.01同设置对照。最佳静态为EMA0.5；固定次要对照包含同EMA0.8。30名此前已观察的非控制器受试者、单个分类器种子17，属于探索性迁移；不根据目标表现重新选择模型或步长。</p>'+f'<p>已完成{finished}/30人、{len(rows)}条方法预测流审计。每个当前窗口先预测后更新，任务标签只用于离线评分。</p>'
if complete:
 comparisons=[("fitted_selected",lock["source_selection"]["best_static"],"primary_vs_source_best_static"),("fitted_selected","frozen","vs_own_frozen"),("fitted_selected_ema08","weak_ema","matched_weakEMA"),("fitted_selected","unfitted_selected","vs_source_selected_unfitted"),("fitted_selected","unfitted_matched","vs_sameLR_unfitted"),("fitted_selected","entropy_selected","vs_source_selected_entropy")];out=[]
 for cohort in ["all","young","middle","older"]:
  g=d if cohort=="all" else d[d.cohort==cohort];q=g.pivot(index="subject",columns="method",values="f1")
  for method,ref,label in comparisons:
   delta=(q[method]-q[ref]).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta))).mean(1),[.025,.975]);out.append(dict(cohort=cohort,comparison=label,n_people=len(delta),mean_method_f1=q[method].mean(),mean_reference_f1=q[ref].mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],positive_people=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_person_pp=delta.min()))
 comp=pd.DataFrame(out);comp.to_csv(o/"paired_comparisons.csv",index=False);means=d.groupby("method").mean(numeric_only=True).drop(columns="subject").reset_index();means.to_csv(o/"method_means.csv",index=False);html+='<h2>按受试者配对</h2><p>区间按受试者bootstrap，未涵盖训练种子及反复探索选择的不确定性。每人等权；平均改善不能排除个体损害。</p><section>'+comp[comp.cohort=="all"].to_html(index=False,float_format=lambda x:f"{x:.5f}")+'</section><h2>损失及分类改变</h2><section>'+means.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'</section>';print(comp[comp.cohort=="all"].to_string(index=False))
html+='<p><a href="participants.csv">逐人结果</a> · <a href="audit.json">独立审计</a> · <a href="selection_lock.json">源选择和模型哈希</a></p></html>';(o/"report.html").write_text(html);print("COMPLETE",complete,finished,len(rows))

from common import *
import pandas as pd
out=ROOT/"phase25";rows=[];audit=0
for f in sorted(out.glob("*/s*.json")):
 d=json.loads(f.read_text());saved=np.load(f.with_suffix(".npz"));m=metrics(saved["y"],saved["predictions"]);assert abs(m["macro_f1"]-d["metrics"]["macro_f1"])<1e-12;assert d["nonwritable_state_unchanged"];audit+=1;rows.append(dict(source=d["source"],policy=d["policy"],subject=d["subject"],f1=100*m["macro_f1"],ce=m["ce"],commit_fraction=d["commit_fraction"]))
for source in ["young","mixed"]:
 for sid in [5,15,25,35]:
  r=json.loads((out/f"{source}_future435"/f"s{sid:02d}.json").read_text())
  for policy,key in [("frozen","frozen"),("weak_ema","weak_ema")]:rows.append(dict(source=source,policy=policy,subject=sid,f1=100*r[key]["macro_f1"],ce=r[key]["ce"],commit_fraction=0.))
  r=json.loads((ROOT/"phase22"/f"{source}_entropy_0.001"/f"s{sid:02d}.json").read_text());rows.append(dict(source=source,policy="continuous",subject=sid,f1=100*r["adapted"]["macro_f1"],ce=r["adapted"]["ce"],commit_fraction=1.))
 d=json.loads((out/f"{source}_static_matched/complete.json").read_text());assert d["best_val_f1"]==max([d["initial_val_f1"]]+[v["val_f1"] for v in d["history"]]);assert d["nonwritable_state_unchanged"] and d["radius_checked"]
 for r in d["results"]:
  saved=np.load(out/f"{source}_static_matched"/f"s{r['subject']:02d}.npz");m=metrics(saved["y"],saved["predictions"]);assert abs(m["macro_f1"]-r["metrics"]["macro_f1"])<1e-12;audit+=1;rows.append(dict(source=source,policy="static_matched",subject=r["subject"],f1=100*m["macro_f1"],ce=m["ce"],commit_fraction=0.))
 for sid,indices in d["labels_used"].items():
  expected=np.unique(np.concatenate([np.load(ROOT/"phase24"/f"{source}_{trajectory}_s{int(sid):02d}/pulses.npz")["future_labels_used"] for trajectory in ["continuous","reset64"]]));assert np.array_equal(expected,np.array(indices))
frame=pd.DataFrame(rows);assert len(frame)==80 and audit==56;frame.to_csv(out/"participant_metrics.csv",index=False);summary=frame.groupby(["source","policy"])[["f1","ce","commit_fraction"]].mean().reset_index();summary.to_csv(out/"summary.csv",index=False);decision=[]
for source,g in summary.groupby("source"):
 q=g.set_index("policy");best_simple=q.loc[["frozen","weak_ema","continuous","cadence2","cadence4"],"f1"].max();decision.append(dict(source=source,future435_f1=float(q.loc["future435","f1"]),current435_f1=float(q.loc["current435","f1"]),best_simple_f1=float(best_simple),future435_beats_simple=bool(q.loc["future435","f1"]>best_simple),future435_beats_current=bool(q.loc["future435","f1"]>q.loc["current435","f1"])))
(out/"source_decision.json").write_text(json.dumps(dict(source_validation_only=True,decisions=decision,no_target_gate_evaluation_launched=True),indent=2));(out/"audit.json").write_text(json.dumps(dict(status="PASS",recomputed_new_prediction_streams=audit,matched_future_label_sets_exact=True,static_epoch0_fallback_selection_correct=True,runtime_radius_and_nonwritable_assertions=True,proposal_graph_equivalence_audit="proposal_forward_audit.json"),indent=2))
html="<!doctype html><html lang=zh><meta charset=utf-8><title>源域真实保留策略与静态校准</title><style>body{max-width:1200px;margin:40px auto;font:16px/1.7 system-ui}td,th{padding:6px;border-bottom:1px solid #ddd}</style><h1>源验证：真实保留策略与同标签静态校准</h1><p>48个人策略配置及2组静态校准全部完成。仅源验证5、15、25、35四人，不是目标泛化结果。写入策略只选KEEP/COMMIT，每窗先预测后决策；阈值0、lr=.001。固定每2/4窗口更新作为更新频率对照。静态校准使用与未来收益预测器完全相同的2168个源控制器未来窗口标签，仅训练同一LayerNorm范围，源验证选epoch且包含冻结回退。</p>"+summary.to_html(index=False,float_format=lambda x:f"{x:.6f}",border=0)+"<p>未来435维策略在年轻来源下略胜简单输出平滑，但仍低于每两窗更新和连续更新；同容量当前收益策略非常接近。源域可观察性的小正值没有转化为超过强简单方法的明确优势。静态标签预算已严格匹配，不能把额外监督带来的潜在效果自动归因于在线保留。</p><p>56条新增预测流复算通过，静态校准的标签集合、epoch0回退及参数约束检查通过。主比较应同时看F1和CE，不能把小的反事实CE变化直接当作Macro-F1提升。</p><p><a href=source_decision.json>源域逐来源比较</a> · <a href=audit.json>审计</a> · <a href=protocol.json>协议</a> · <a href=../phase24/report.html>留人反事实诊断</a></p></html>"
(out/"report.html").write_text(html);print(summary.to_string(index=False));print(json.dumps(decision,indent=2));print("AUDIT PASS",audit)

from common import *
import pandas as pd
out=ROOT/"phase22";expected=20 if (out/"extension_protocol.json").exists() else 12;rows=[];audited=0;max_error=0.
for f in sorted(out.glob("*/complete.json")):
 d=json.loads(f.read_text())
 if d.get("benchmark_limit",0):continue
 assert len(d["results"])==4
 for r in d["results"]:
  sid=r["subject"];saved=np.load(f.parent/f"s{sid:02d}.npz");y=saved["y"];data=ROOT/("data/processed" if sid<20 else "phase2/processed");assert np.array_equal(y,np.load(data/f"s{sid:02d}_meta.npz")["y"])
  for method in ["frozen","adapted","weak_ema"]:
   p=saved[method];assert p.shape==(len(y),5) and np.isfinite(p).all();m=metrics(y,p);max_error=max(max_error,abs(m["macro_f1"]-r[method]["macro_f1"]));audited+=1
   rows.append(dict(source=d["source"],objective=d["objective"],lr=d["lr"],subject=sid,method=method,f1=m["macro_f1"],ce=m["ce"],seconds=r["seconds"]))
if not rows:raise SystemExit("No completed source cases")
frame=pd.DataFrame(rows);frame.to_csv(out/"participant_metrics.csv",index=False);summ=[]
for keys,g in frame.groupby(["source","objective","lr"]):
 q=g.pivot(index="subject",columns="method",values="f1");delta=100*(q.adapted-q.frozen);summ.append(dict(source=keys[0],objective=keys[1],lr=keys[2],frozen=100*q.frozen.mean(),adapted=100*q.adapted.mean(),weak_ema=100*q.weak_ema.mean(),delta_pp=delta.mean(),minus_ema_pp=100*(q.adapted-q.weak_ema).mean(),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min()))
s=pd.DataFrame(summ);s.to_csv(out/"summary.csv",index=False);selections=[]
if len(s)==expected:
 for (source,objective),g in s.groupby(["source","objective"]):
  best=g.loc[g.adapted.idxmax()];selections.append(dict(source=source,objective=objective,lr=0. if best.adapted<=best.frozen else float(best.lr),positive_lr=float(best.lr),selected_f1=max(best.frozen,best.adapted),weak_ema=float(best.weak_ema)))
 (out/"source_selection.json").write_text(json.dumps(dict(source_validation_only=True,selections=selections),indent=2))
assert max_error<1e-12;(out/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",complete_cases=len(s),prediction_streams=audited,max_f1_recompute_error=max_error,labels_match_original=True,scope="Saved-stream numerical audit; batch1/source-causality audit saved separately"),indent=2))
html="<!doctype html><html lang=zh><meta charset=utf-8><title>充分训练模型的源域适应</title><style>body{max-width:1200px;margin:40px auto;font:16px/1.7 system-ui}td,th{padding:6px;border-bottom:1px solid #ddd}</style><h1>充分训练模型的源域适应筛选</h1><p>已完成 "+str(len(s))+" /"+str(expected)+"配置。GPU运行，CPU8–23核；固定源域完整训练模型，仅LayerNorm仿射可更新。当前30秒先预测再更新，不使用目标标签。比较熵与固定源模型的因果时间教师，初始学习率1e−6、1e−5、1e−4；源验证最优点位于上边界，随后扩展.001、.01（另存扩展协议）。这里四位源验证人员用于选择，不能作为泛化效果估计。</p>"+s.to_html(index=False,float_format=lambda x:f"{x:.6f}",border=0)+"<p>另有完整逐窗口冻结审计：两种来源×四位人员，批量缓存与逐窗口分类0处差异，未来样本扰动不改变当前预测。</p><p><a href=protocol.json>协议</a> · <a href=prediction_audit.json>指标审计</a> · <a href=../phase21/report.html>完整模型训练</a></p></html>"
(out/"report.html").write_text(html);print(s.to_string(index=False));print(json.dumps(selections,indent=2))

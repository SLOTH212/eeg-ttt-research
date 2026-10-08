from common import *
import pandas as pd
out=ROOT/"phase21";rows=[];audits=[]
for f in sorted((out/"evaluation").glob("*/evaluation.json")):
 d=json.loads(f.read_text());source=d["source"];kind=d["kind"]
 selection=d["selection"];assert selection["selection"]["best_val_f1"]==max(c["best_val_f1"] for c in selection["candidates"])
 ck=torch.load(out/selection["selected_directory"]/"model.pt",map_location="cpu",weights_only=False);assert ck["best_val_f1"]==max([ck["initial_val_f1"]]+[h["val_f1"] for h in ck["history"]])
 for r in d["results"]:
  sid=r["subject"];saved=np.load(f.parent/f"s{sid:02d}.npz");base=np.load(ROOT/"phase18"/f"{source}_{kind}"/f"s{sid:02d}.npz");assert np.array_equal(saved["y"],base["y"]);y=saved["y"];p=saved["source_tuned"];m=metrics(y,p);assert abs(m["macro_f1"]-r["macro_f1"])<1e-12
  for method,pr in [("frozen",base["frozen"]),("full_source_tuned",p)]:
   assert np.isfinite(pr).all() and np.max(np.abs(pr.sum(1)-1))<1e-5
   mm=metrics(y,pr);rows.append(dict(source=source,kind=kind,subject=sid,cohort=r["cohort"],method=method,macro_f1=mm["macro_f1"],ce=mm["ce"],kappa=mm["kappa"]))
  audits.append(dict(source=source,kind=kind,subject=sid,metrics_match=True,labels_match=True))
if not rows:raise SystemExit("No completed target evaluations yet")
frame=pd.DataFrame(rows);frame.to_csv(out/"participant_metrics.csv",index=False);summ=[]
for keys,g in frame.groupby(["source","kind","cohort"]):
 q=g.pivot(index="subject",columns="method",values="macro_f1");delta=100*(q.full_source_tuned-q.frozen).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);summ.append(dict(source=keys[0],kind=keys[1],cohort=keys[2],n=len(delta),frozen=100*q.frozen.mean(),full_source_tuned=100*q.full_source_tuned.mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],wins=int((delta>0).sum()),harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min()))
s=pd.DataFrame(summ);s.to_csv(out/"summary.csv",index=False)
# Matched supervised pretrained-vs-random contrast, only once both are complete.
contrasts=[]
for (source,cohort),g in frame[frame.method=="full_source_tuned"].groupby(["source","cohort"]):
 q=g.pivot(index="subject",columns="kind",values="macro_f1")
 if not {"pretrained","random"}<=set(q.columns):continue
 assert not q.isna().any().any();delta=100*(q.pretrained-q.random).to_numpy();rng=np.random.default_rng(20261006);ci=np.quantile(rng.choice(delta,(20000,len(delta)),replace=True).mean(1),[.025,.975]);contrasts.append(dict(source=source,cohort=cohort,n=len(delta),pretrained_minus_random_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1]))
c=pd.DataFrame(contrasts);c.to_csv(out/"pretraining_contrast.csv",index=False)
(out/"prediction_audit.json").write_text(json.dumps(dict(status="PASS",completed_carriers=len(list((out/"evaluation").glob("*/evaluation.json"))),target_prediction_streams=len(audits),checks="Saved probabilities, labels, F1 recomputation, source-only lr and epoch selection",details=audits),indent=2))
html="<!doctype html><html lang=zh><meta charset=utf-8><title>GPU全编码器训练</title><style>body{max-width:1250px;margin:40px auto;font:16px/1.7 system-ui}td,th{padding:6px;border-bottom:1px solid #ddd}</style><h1>GPU全编码器源域监督训练</h1><p>RTX5090，CPU8–23核；约582万参数全部可训练。预训练/随机初始化分别训练，学习率和epoch仅由四位源验证人员选择，含冻结回退。以下为已观察过的人群上的探索性评估，不是新盲测，不是在线TTA。一个种子，区间按人配对bootstrap。</p>"+s.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<img src=comparison.png style='max-width:100%' alt='配对结果图'><h2>同架构完整训练后的预训练贡献</h2>"+c.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<p><a href=protocol.json>预先保存协议</a> · <a href=prediction_audit.json>预测审计</a> · <a href=../phase18/report.html>此前局部训练对照</a> · <a href=strong_controls.html>既有强基线比较</a></p></html>"
(out/"report.html").write_text(html);print(s.to_string(index=False));print(c.to_string(index=False))

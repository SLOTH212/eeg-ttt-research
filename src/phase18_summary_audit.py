from common import *
import pandas as pd
out=ROOT/"phase18";rows=[];summaries=[];audited=0;max_error=0.
for path in sorted(out.glob("*/evaluation.json")):
 d=json.loads(path.read_text());ck=torch.load(path.parent/"model.pt",map_location="cpu",weights_only=False)
 assert ck["best_val_f1"]>=ck["initial_val_f1"]
 expected=max([ck["initial_val_f1"]]+[h["val_f1"] for h in ck["history"]]);assert abs(ck["best_val_f1"]-expected)<1e-12
 for r in d["results"]:
  saved=np.load(path.parent/f"s{r['subject']:02d}.npz");pr=saved[r["method"]];y=saved["y"];assert pr.shape==(len(y),5) and np.isfinite(pr).all();assert np.max(np.abs(pr.sum(1)-1))<1e-5
  sid=r["subject"];data=ROOT/("data/processed" if sid<20 else "phase2/processed" if sid<40 else "phase6/processed");assert np.array_equal(y,np.load(data/f"s{sid:02d}_meta.npz")["y"])
  m=metrics(y,pr)
  for key in ["macro_f1","balanced_accuracy","ce"]:max_error=max(max_error,abs(m[key]-r[key]))
  assert np.array_equal(m["confusion"],r["confusion"]);audited+=1
  rows.append(dict(source=d["source"],kind=d["kind"],subject=sid,cohort=r["cohort"],method=r["method"],macro_f1=m["macro_f1"],best_epoch=ck["best_epoch"]))
frame=pd.DataFrame(rows);frame.to_csv(out/"participant_metrics.csv",index=False)
for keys,g in frame.groupby(["source","kind","cohort"]):
 pivot=g.pivot(index="subject",columns="method",values="macro_f1");delta=100*(pivot.source_tuned-pivot.frozen).to_numpy();rng=np.random.default_rng(20261006);bs=rng.choice(delta,(20000,len(delta)),replace=True).mean(1);ci=np.quantile(bs,[.025,.975]);summaries.append(dict(source=keys[0],kind=keys[1],cohort=keys[2],n=len(delta),frozen=100*pivot.frozen.mean(),source_tuned=100*pivot.source_tuned.mean(),delta_pp=delta.mean(),ci_low=ci[0],ci_high=ci[1],harm_over_1pp=int((delta< -1).sum()),worst_delta_pp=delta.min()))
summary=pd.DataFrame(summaries);summary.to_csv(out/"summary.csv",index=False);assert audited==272 and max_error<1e-10
(out/"final_audit.json").write_text(json.dumps(dict(status="PASS",prediction_streams=audited,max_metric_error=max_error,source_epoch_selection_checked=True,raw_labels_match=True,scope="Saved prediction and epoch-selection audit; not independent retraining or full frozen-parameter audit"),indent=2))
(out/"report.html").write_text("<!doctype html><html lang=zh><meta charset=utf-8><title>源域局部监督微调</title><style>body{max-width:1200px;margin:35px auto;font:16px/1.7 system-ui}td,th{padding:6px;border-bottom:1px solid #ddd}</style><h1>源域局部监督微调：四组全部完成</h1><p>仅训练最后两层Transformer的LayerNorm及分类头，其他权重冻结。预训练和随机初始化分别在年轻来源、混合来源上训练；源验证选择包含epoch0冻结回退。此为离线源域监督训练，不是TTA。仅seed17，以下人群已参与此前探索，不能视为独立盲测。</p>"+summary.to_html(index=False,float_format=lambda x:f"{x:.3f}",border=0)+"<p>区间为按人配对bootstrap95%区间。未见跨人群一致提升，不能据此否定更充分的编码器训练。GPU授权后的全编码器实验另存phase21。</p><p>272条预测流指标复算通过；原始标签一致，源验证epoch选择正确。</p><a href=final_audit.json>审计范围与结果</a> · <a href=../phase21/protocol.json>GPU扩展协议</a></html>")
print(summary.to_string(index=False));print("AUDIT PASS",audited,max_error)

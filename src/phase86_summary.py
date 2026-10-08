from common import *
import pandas as pd,hashlib
O=ROOT/"phase86";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();audit=json.loads((O/"implementation_audit.json").read_text());assert audit["status"]=="PASS"
rows=[];budget={};orders=[];headstates=[]
assert not set(pr["meta_subjects"]) & set(pr["controllers"]+pr["held_source_validation"])
assert not set(pr["controllers"]) & set(pr["held_source_validation"])
for which in ["future","current","reconstruction"]:
 d=json.loads((O/which/"complete.json").read_text());assert d["classifier_unchanged_exact"] and d["epochs"]==5 and d["pulses"]==317
 used={}
 for sid in pr["meta_subjects"]:
  times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"];labels=set()
  for t in times:labels.update([] if which=="reconstruction" else [int(t)] if which=="current" else range(int(t)+1,int(t)+5))
  used[str(sid)]=sorted(labels)
 assert d["labels_used"]==used and d["label_count"]==sum(map(len,used.values()))
 order=json.loads((O/which/"training_orders.json").read_text())["orders"];gen=torch.Generator(device="cpu").manual_seed(17);reference=[torch.randperm(317,generator=gen).tolist() for _ in range(5)];assert order==reference;orders.append(order)
 h=torch.load(O/which/"head.pt",map_location="cpu",weights_only=False);assert set(h["head"])=={"weight","bias"} and h["head"]["weight"].shape==(200,200) and h["head"]["bias"].shape==(200,);assert all(torch.isfinite(v).all() for v in h["head"].values());headstates.append(h["head"]);budget[which]=dict(label_count=d["label_count"],pulses=d["pulses"],label_presentations=d["label_count"]*5,seconds=d["seconds"],head_sha256=hashlib.sha256((O/which/"head.pt").read_bytes()).hexdigest())
 for r in d["history"]:rows.append(dict(target=which,**r))
assert orders[0]==orders[1]==orders[2]
for i in range(3):
 for j in range(i):assert not torch.equal(headstates[i]["weight"],headstates[j]["weight"])
out=dict(status="PASS",three_distinct_heads=True,source_splits_disjoint=True,labels_independently_reconstructed=True,dedicated_rng_orders_replayed_exact=True,five_fixed_epochs=True,budget=budget,implementation_max_relative_error=max(r["relative_error"] for r in audit["finite_differences"]),only_auxiliary_head_saved=True,classifier_unchanged_every_epoch_asserted=True)
(O/"training_audit.json").write_text(json.dumps(out,indent=2));df=pd.DataFrame(rows);df.to_csv(O/"training_history.csv",index=False)
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>CBraMod辅助波形头元训练</h1><p>固定主分类器及编码器初始权重，只训练原有200→200线性波形投影。内层在当前无标签重建上虚拟写入LayerNorm参数，外层利用源人群的后续4段或当前段分类损失学习辅助头；另有仅重建继续训练对照。额外分类监督成本不可忽略。</p>'+pd.DataFrame(budget).T.to_html()+'<h2>固定五轮训练记录</h2>'+df.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p>训练目标降低不是写入成功证据，重建MSE上升也可能暴露目标冲突。下一阶段必须同时验证源人群实际任务收益、等范数对照和重建有效性。三个分支严格采用相同317脉冲和顺序，计算成本不同：仅重建没有二阶任务梯度。</p><p><a href="training_audit.json">训练与标签审计</a> · <a href="implementation_audit.json">超梯度有限差分</a> · <a href="protocol.json">固定方案</a> · <a href="../phase87/report.html">实际写入验证（完成后）</a></p>')
print(out)

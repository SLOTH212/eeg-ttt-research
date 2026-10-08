from common import *
from phase2_temporal import ema
import pandas as pd
o=ROOT/"phase65";assert (o/"training_complete.json").exists();protocol=json.loads((o/"protocol.json").read_text())
base=torch.load(ROOT/"phase28/mixed_joint_0/model.pt",map_location="cpu",weights_only=False)["state"];layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text())["parameters"];writable={p["name"] for p in layout};rows=[];checks=[];orders={}
for seed in [17,29,43]:
 folder=ROOT/"phase53" if seed==17 else o/f"seed{seed}"
 for target in ["current","future"]:
  out=folder/("static_"+target);meta=json.loads((out/"complete.json").read_text());state=torch.load(out/"model.pt",map_location="cpu",weights_only=False)["state"];reference=json.loads((ROOT/"phase52"/target/"complete.json").read_text())
  assert meta["labels_used"]==reference["labels_used"] and meta["label_count"]==protocol["labels"][target]
  assert meta["nonwritable_state_unchanged"] and meta["radius_checked"] and len(meta["history"])==20
  assert all(torch.equal(v,state[n]) for n,v in base.items() if n not in writable)
  maxratio=0.
  for n in writable:
   v=base[n];radius=.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5;ratio=float((state[n]-v).norm())/radius;maxratio=max(maxratio,ratio);assert ratio<=1.00001
  options=[dict(epoch=0,val_f1=meta["initial_val_f1"])]+meta["history"];best=max(options,key=lambda q:q["val_f1"]);assert best["epoch"]==meta["best_epoch"] and best["val_f1"]==meta["best_val_f1"]
  if seed!=17:
   assert meta["calibration_training_seed"]==seed
   order=json.loads((out/"training_orders.json").read_text());g=torch.Generator().manual_seed(seed);expected=[torch.randperm(meta["label_count"],generator=g).tolist() for _ in range(20)];assert order["orders"]==expected;orders[seed,target]=expected
  for sid in [5,15,25,35]:
   z=np.load(out/f"s{sid:02d}.npz");y=z["y"];p=z["predictions"];lr={"current":.0001,"future":.01}[target];adaptfolder=(ROOT/"phase55" if seed==17 else ROOT/"phase61"/f"seed{seed}")/f"{target}_{lr:g}";adapt=np.load(adaptfolder/f"s{sid:02d}.npz");frozen=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")
   assert np.array_equal(y,adapt["y"]) and np.array_equal(y,frozen["y"]);assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5
   mm=metrics(y,p);expectedscore=next(x["metrics"]["macro_f1"] for x in meta["results"] if x["subject"]==sid);assert abs(mm["macro_f1"]-expectedscore)<1e-12
   for smooth in [1.,.8,.5]:
    probs=[adapt["p"],p,frozen["p"]];scores=[100*metrics(y,q if smooth==1 else ema(q,smooth))["macro_f1"] for q in probs]
    rows.append(dict(seed=seed,target=target,subject=sid,ema=smooth,adaptive_f1=scores[0],static_f1=scores[1],frozen_f1=scores[2],adaptive_minus_static_pp=scores[0]-scores[1],static_minus_frozen_pp=scores[1]-scores[2],best_static_epoch=meta["best_epoch"]))
  checks.append(dict(seed=seed,target=target,label_count=meta["label_count"],best_epoch=meta["best_epoch"],max_radius_fraction=maxratio,nonwritable_exact=True))
for target in ["current","future"]:assert orders[29,target]!=orders[43,target]
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False)
ss=df.groupby(["seed","target","ema"])[["adaptive_f1","static_f1","frozen_f1","adaptive_minus_static_pp","static_minus_frozen_pp"]].mean();ss.to_csv(o/"seed_summary.csv")
people=df.groupby(["subject","target","ema"])[["adaptive_f1","static_f1","frozen_f1","adaptive_minus_static_pp","static_minus_frozen_pp"]].mean();people.to_csv(o/"seedmean_people.csv")
summary=people.groupby(["target","ema"]).mean();summary.to_csv(o/"seedmean_summary.csv")
audit=dict(status="PASS",calibration_checkpoints=6,new_calibrations=4,seed_values=[17,29,43],base_classifier_seed=17,exact_label_indices_verified=True,all_source_epoch_selections_verified=True,independent_source_radius_checked=True,nonwritable_state_exact=True,actual_epoch_permutations_verified=True,per_person_seed_means_no_ensemble=True,checks=checks)
(o/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>双方训练顺序重复后的等标签比较</h1><p>阶段61已重复辅助头优化顺序，本阶段同样重复静态校准顺序29/43，17沿用阶段53。各自使用完全相同317/1268标签索引；基础分类器固定seed17。静态算法保持20轮源侧最佳原始Macro-F1轮次含0的旧规则，辅助头保持最终第5轮和阶段55选定在线步长。两者原有选择规则不完全对称，明确保留；没有新调参网格。</p><h2>每名受试者先平均三个种子，再平均四人</h2>'+summary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>逐种子</h2>'+ss.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>每人的种子均值</h2>'+people.to_html(float_format=lambda x:f"{x:.5f}")+'<p>这检验固定基础分类器下两种训练算法的顺序敏感性，不是独立分类器种子，也不是概率集成。源四人已反复使用且参与选择，不把这里的差距当作盲测泛化估计。本阶段不自动晋级目标人群。静态与在线同样列出EMA1/.8/.5，不跨不同标签预算混称公平比较。</p><p><a href="protocol.json">预定协议</a> · <a href="audit.json">独立审计</a> · <a href="code_provenance.json">代码改动范围</a></p></html>';(o/"report.html").write_text(html)
print(summary.to_string());print(ss.to_string());print(audit)

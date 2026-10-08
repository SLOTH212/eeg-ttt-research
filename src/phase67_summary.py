from common import *
from phase2_temporal import ema
import pandas as pd
o=ROOT/"phase67";assert (o/"training_complete.json").exists();protocol=json.loads((o/"protocol.json").read_text())
base=torch.load(ROOT/"phase28/mixed_joint_0/model.pt",map_location="cpu",weights_only=False)["state"];layout=json.loads((ROOT/"phase48/parameter_layout.json").read_text())["parameters"];writable={p["name"] for p in layout};rows=[];checks=[];orders={}
for seed in [17,29,43]:
 folder=o/f"seed{seed}"
 for target in ["current","future"]:
  out=folder/("static_"+target);meta=json.loads((out/"complete.json").read_text());state=torch.load(out/"model.pt",map_location="cpu",weights_only=False)["state"];reference=json.loads((ROOT/"phase52"/target/"complete.json").read_text())
  assert meta["labels_used"]==reference["labels_used"] and meta["label_count"]==protocol["unique_labels"][target]
  assert meta["nonwritable_state_unchanged"] and meta["radius_checked"] and len(meta["history"])>=5
  assert all(torch.equal(v,state[n]) for n,v in base.items() if n not in writable)
  maxratio=0.
  for n in writable:
   v=base[n];radius=.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5;ratio=float((state[n]-v).norm())/radius;maxratio=max(maxratio,ratio);assert ratio<=1.00001
  options=[dict(epoch=0,val_f1=meta["initial_val_f1"])]+meta["history"];best=max(options,key=lambda q:q["val_f1"]);assert best["epoch"]==meta["best_epoch"] and best["val_f1"]==meta["best_val_f1"]
  assert meta.get("evaluation_epoch",meta["best_epoch"])==5
  original_folder=(ROOT/"phase53" if seed==17 else ROOT/"phase65"/f"seed{seed}")/("static_"+target)
  original_meta=json.loads((original_folder/"complete.json").read_text())
  for new,old in zip(meta["history"][:5],original_meta["history"][:5]):
   assert new["val_f1"]==old["val_f1"] and new["participants"]==old["participants"]
  if not out.is_symlink():
   assert meta["calibration_training_seed"]==seed and len(meta["history"])==5
  if (out/"training_orders.json").exists():
   order=json.loads((out/"training_orders.json").read_text());g=torch.Generator().manual_seed(seed);expected=[torch.randperm(meta["label_count"],generator=g).tolist() for _ in range(len(order["orders"]))];assert order["orders"]==expected;orders[seed,target]=expected[:5]
  for sid in [5,15,25,35]:
   z=np.load(out/f"s{sid:02d}.npz");y=z["y"];p=z["predictions"];lr={"current":.0001,"future":.01}[target];adaptfolder=(ROOT/"phase55" if seed==17 else ROOT/"phase61"/f"seed{seed}")/f"{target}_{lr:g}";adapt=np.load(adaptfolder/f"s{sid:02d}.npz");frozen=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")
   assert np.array_equal(y,adapt["y"]) and np.array_equal(y,frozen["y"]);assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5
   mm=metrics(y,p);expectedscore=next(x["metrics"]["macro_f1"] for x in meta["results"] if x["subject"]==sid);assert abs(mm["macro_f1"]-expectedscore)<1e-12
   for smooth in [1.,.8,.5]:
    probs=[adapt["p"],p,frozen["p"]];scores=[100*metrics(y,q if smooth==1 else ema(q,smooth))["macro_f1"] for q in probs]
    rows.append(dict(seed=seed,target=target,subject=sid,ema=smooth,adaptive_f1=scores[0],static_f1=scores[1],frozen_f1=scores[2],adaptive_minus_static_pp=scores[0]-scores[1],static_minus_frozen_pp=scores[1]-scores[2],static_evaluation_epoch=5,recorded_source_best_epoch=meta["best_epoch"]))
  checks.append(dict(seed=seed,target=target,label_count=meta["label_count"],best_epoch=meta["best_epoch"],max_radius_fraction=maxratio,nonwritable_exact=True))
for target in ["current","future"]:assert orders[29,target]!=orders[43,target]
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False)
ss=df.groupby(["seed","target","ema"])[["adaptive_f1","static_f1","frozen_f1","adaptive_minus_static_pp","static_minus_frozen_pp"]].mean();ss.to_csv(o/"seed_summary.csv")
people=df.groupby(["subject","target","ema"])[["adaptive_f1","static_f1","frozen_f1","adaptive_minus_static_pp","static_minus_frozen_pp"]].mean();people.to_csv(o/"seedmean_people.csv")
summary=people.groupby(["target","ema"]).mean();summary.to_csv(o/"seedmean_summary.csv")
audit=dict(status="PASS",calibration_checkpoints=6,new_calibrations=4,reused_exact_epoch5=2,fixed_epoch5=True,original_twenty_epoch_prefix_scores_exact=True,class_label_presentations={"current":1585,"future":6340},seed_values=[17,29,43],base_classifier_seed=17,exact_label_indices_verified=True,historical_selection_metadata_verified=True,evaluation_epoch_fixed5=True,independent_source_radius_checked=True,nonwritable_state_exact=True,actual_epoch_permutations_verified=True,per_person_seed_means_no_ensemble=True,checks=checks)
(o/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>固定五轮、匹配类别标签呈现次数的静态对照</h1><p>这是阶段65后的比较公平性检查。静态校准也固定最终第5轮，不选最佳轮；当前317标签各用5次共1585次，未来1268标签各用5次共6340次，与辅助头五轮元训练的分类外层标签呈现次数一致。其余静态设置不变。当前seed17/29已有第5轮检查点，严格复用；其余四组重跑五轮，每轮源验证分数与旧20轮训练前缀完全一致。</p><p>不宣称计算量或所有算法因素匹配：元训练使用二阶梯度，静态用一阶；优化参数角色和类别权重不同。原20轮源最佳静态结果保留。这里只比较固定第五轮状态，并非新的盲测或预先独立确认检验。</p><h2>先在人内平均三个优化顺序种子，再平均人</h2>'+summary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>逐种子</h2>'+ss.to_html(float_format=lambda x:f"{x:.5f}")+'<p>基础分类器仍固定seed17，在线步长仍按阶段55固定，输出EMA网格双方相同。无概率集成，不把种子当独立受试者；源四人已反复使用。不自动推进目标人群。</p><p><a href="protocol.json">检查方案</a> · <a href="audit.json">独立审计</a> · <a href="../phase65/report.html">原20轮选择结果</a></p></html>'
(o/"report.html").write_text(html)
print(summary.to_string());print(ss.to_string());print(audit)

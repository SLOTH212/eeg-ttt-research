from common import *
from phase2_temporal import ema
import pandas as pd,pickle,hashlib
o=ROOT/"phase66";pr=json.loads((o/"protocol.json").read_text());assert (o/"training_complete.json").exists()
rows=[];checks=[];selected={}
for kind in pr["kinds"]:
 candidates=[]
 for lr in pr["encoder_lrs"][kind]:
  folder=o/f"{kind}_{lr:g}";m=json.loads((folder/"complete.json").read_text());ck=torch.load(folder/"model.pt",map_location="cpu",weights_only=False);st=ck["state"]
  assert len(m["history"])==10 and m["training_subjects"]==pr["training_subjects"] and m["source_validation_only"] and m["initial_probe_argmax_exact"] and m["reconstruction_projection_unchanged"]
  opts=[dict(epoch=0,val_f1=m["initial_val_f1"])]+m["history"];best=max(opts,key=lambda x:x["val_f1"]);assert best["epoch"]==m["best_epoch"]==ck["best_epoch"] and best["val_f1"]==m["best_val_f1"]
  orig=torch.load(ROOT/"phase63"/(kind+"_encoder_state.pt"),map_location="cpu",weights_only=True)
  assert all(torch.equal(v,st["encoder."+n]) for n,v in orig.items() if n.startswith("proj_out.") or n=="patch_embedding.mask_encoding")
  with (ROOT/"phase63"/("probe_"+kind)/"model.pkl").open("rb") as f:probe=pickle.load(f)
  assert np.array_equal(st["mean"].numpy(),probe["scaler"].mean_.astype(np.float32)) and np.array_equal(st["scale"].numpy(),probe["scaler"].scale_.astype(np.float32))
  if m["best_epoch"]==0:
   assert all(torch.equal(v,st["encoder."+n]) for n,v in orig.items());assert np.array_equal(st["head.weight"].numpy(),probe["model"].coef_.astype(np.float32))
  saved=json.loads((folder/"training_orders.json").read_text());g=torch.Generator().manual_seed(17);expected=[torch.randperm(m["training_valid_windows"],generator=g).tolist() for _ in range(10)];assert saved["orders"]==expected
  for sid in pr["validation_subjects"]:
   z=np.load(folder/f"s{sid:02d}.npz");ref=np.load(ROOT/"phase63"/("probe_"+kind)/f"s{sid:02d}.npz");assert np.array_equal(z["y"],ref["y"]) and np.isfinite(z["p"]).all()
   raw=metrics(z["y"],z["p"]);stored=next(x["metrics"] for x in m["results"] if x["subject"]==sid);assert abs(raw["macro_f1"]-stored["macro_f1"])<1e-12
   for smooth in [1.,.8,.5]:
    mm=metrics(z["y"],z["p"] if smooth==1 else ema(z["p"],smooth));rows.append(dict(subject=sid,kind=kind,lr=lr,ema=smooth,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],cross_entropy=mm["ce"]))
  candidates.append(dict(kind=kind,lr=lr,folder=str(folder),macro_f1=100*m["best_val_f1"],best_epoch=m["best_epoch"],initial_f1=100*m["initial_val_f1"],checkpoint_sha256=hashlib.sha256((folder/"model.pt").read_bytes()).hexdigest()))
  checks.append(dict(kind=kind,lr=lr,best_epoch=m["best_epoch"],source_selection_verified=True,projection_unchanged=True,initial_probe_exact_classes=True,actual_order_verified=True))
 selected[kind]=max(candidates,key=lambda q:q["macro_f1"])
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False);summary=df.groupby(["kind","lr","ema"])[["macro_f1","balanced_accuracy","cross_entropy"]].mean();summary.to_csv(o/"summary.csv")
baselines=pd.read_csv(ROOT/"phase63/summary.csv");comparisons=[]
for kind,candidate in selected.items():
 for smooth in [1.,.8,.5]:
  value=summary.loc[(kind,candidate["lr"],smooth),"macro_f1"]
  for name in ["cbramod_"+kind,"labram_frozen_pretrained","labram_source_finetuned_reference"]:
   reference=float(baselines[(baselines.model==name)&(baselines.ema==smooth)].iloc[0].macro_f1)
   comparisons.append(dict(kind=kind,lr=candidate["lr"],best_epoch=candidate["best_epoch"],ema=smooth,macro_f1=value,reference=name,reference_f1=reference,difference_pp=value-reference))
cmp=pd.DataFrame(comparisons);cmp.to_csv(o/"comparisons.csv",index=False)
(o/"selection.json").write_text(json.dumps(dict(selected=selected,selection="Raw source meanMacroF1,epoch0..10 andtwo fixedrates; do not select by smoothedscore",source_only=True,no_target_advance=True,reconstruction_validity_pending=True),indent=2))
(o/"audit.json").write_text(json.dumps(dict(status="PASS",configs=4,source_validation_trajectories=16,method_streams=48,classifier_seed=17,actual_epoch_order_checked=True,epoch0_fallback_checked=True,reconstruction_projection_exact=True,checks=checks),indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1350px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>CBraMod固定预算源域分类微调</h1><p>两导联30秒、每导联时间平均400维，与阶段63相同预处理和线性头初始化。预训练和随机初始化分别采用阶段21相同两个编码器步长、十轮、AdamW和加权监督预算；源十人训练、源四人选轮次与步长，包含第0轮冻结回退。全程不访问目标人群。仅一个分类训练种子17，原生模型训练模式使用既有dropout。</p><h2>完整网格</h2>'+summary.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>按原始F1选择后的同EMA比较</h2>'+cmp.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<p>这是特定池化头、预处理和固定微调预算的能力检查，不能据此概括CBraMod的最佳性能。原始重建输出头参数完全不变，但编码器变化仍可能破坏重建能力，必须另行验证。分类器训练成绩本身不是测试时适应收益。源四人已经大量使用，不作新盲测声明。</p><p><a href="protocol.json">协议</a> · <a href="selection.json">选择</a> · <a href="audit.json">审计</a></p></html>';(o/"report.html").write_text(html);print(summary.to_string());print(selected);print(cmp.to_string(index=False))

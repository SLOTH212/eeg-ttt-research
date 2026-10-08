from common import *
from phase2_temporal import ema
from sklearn.metrics import f1_score
import pandas as pd,hashlib
o=ROOT/"phase61";pr=json.loads((o/"protocol.json").read_text());assert (o/"evaluation_complete.json").exists()
base=torch.load(ROOT/"phase28/mixed_joint_0/model.pt",map_location="cpu",weights_only=False)["state"]
readout=np.load(ROOT/"phase46/masked_readout.npz")
cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"])
pulse_rows=[];online_rows=[];validity=[];max_error=0.;totalp=0;newonline=0
for seed in [17,29,43]:
 folder=ROOT/"phase52" if seed==17 else o/f"seed{seed}"
 if seed!=17:
  assert json.loads((folder/"graph_audit.json").read_text())["status"]=="PASS"
  assert len(json.loads((folder/"graph_audit.json").read_text())["cases"])==6
  for variant in ["current","future"]:
   saved=json.loads((folder/variant/"training_orders.json").read_text());g=torch.Generator(device="cpu").manual_seed(seed);expected=[torch.randperm(317,generator=g).tolist() for _ in range(5)];assert saved["orders"]==expected and saved["seed"]==seed
  assert json.loads((folder/"current/training_orders.json").read_text())==json.loads((folder/"future/training_orders.json").read_text())
 vd=pd.read_csv(folder/"reconstruction_validity.csv");vd=vd[vd.variant.isin(["current","future"])].copy();vd["seed"]=seed;validity.append(vd)
 for variant in ["current","future"]:
  meta=json.loads((folder/variant/"complete.json").read_text());initialmeta=json.loads((ROOT/"phase52"/variant/"complete.json").read_text())
  state=torch.load(folder/variant/"model.pt",map_location="cpu",weights_only=False)["state"]
  assert meta["epochs"]==5 and meta["labels_used"]==initialmeta["labels_used"] and meta["label_count"]==({"current":317,"future":1268}[variant])
  if seed!=17:assert meta["auxiliary_training_seed"]==seed
  assert all(torch.equal(v,state[n]) for n,v in base.items())
  for key in ["input_mean","input_scale","target_mean","target_scale"]:assert np.array_equal(state["recon_"+key].numpy(),readout[key])
  for sid in [7,17,27,37]:
   d=json.loads((folder/"evaluation"/variant/f"s{sid:02d}.json").read_text());assert d["model_restored_exact"]
   reference=json.loads((ROOT/"phase52/evaluation"/variant/f"s{sid:02d}.json").read_text())
   assert [r["t"] for r in d["rows"]]==[r["t"] for r in reference["rows"]]
   gains=[];bal=[];corrected=0;harmed=0;yy=[];before=[];after=[]
   root=ROOT/("data/processed" if sid<20 else "phase2/processed");truth=np.load(root/f"s{sid:02d}_meta.npz")["y"]
   for rr,ref in zip(d["rows"],reference["rows"]):
    y=np.array(rr["future_labels"]);bp=np.array(rr["future_base_probabilities"]);wp=np.array(rr["future_write_probabilities"])
    assert np.array_equal(y,truth[rr["t"]+1:rr["t"]+5])
    assert abs(bp-np.array(ref["future_base_probabilities"])).max()<2e-5
    assert np.isfinite(wp).all() and abs(wp.sum(1)-1).max()<1e-5
    delta=np.log(wp[np.arange(4),y])-np.log(bp[np.arange(4),y])
    err=max(abs(delta.mean()-rr["future_ce_gain"]),abs((delta*cw[y]).mean()-rr["balanced_future_ce_gain"]));max_error=max(max_error,err);assert err<4e-6
    b=bp.argmax(1);w=wp.argmax(1);corrected+=int(((b!=y)&(w==y)).sum());harmed+=int(((b==y)&(w!=y)).sum());gains.append(delta.mean());bal.append((delta*cw[y]).mean());yy.extend(y);before.extend(b);after.extend(w)
   n=len(yy);pulse_rows.append(dict(seed=seed,variant=variant,subject=sid,pulses=len(gains),future_ce_gain=np.mean(gains),balanced_ce_gain=np.mean(bal),accuracy_gain_pp=100*(corrected-harmed)/n,corrected=corrected,harmed=harmed,sample_f1_gain_pp=100*(f1_score(yy,after,labels=range(5),average="macro",zero_division=0)-f1_score(yy,before,labels=range(5),average="macro",zero_division=0))));totalp+=len(gains)
  lr=pr["online_lr"][variant];runfolder=(ROOT/"phase55" if seed==17 else folder)/f"{variant}_{lr:g}"
  assert (runfolder/"complete.json").exists()
  for sid in [5,15,25,35]:
   z=np.load(runfolder/f"s{sid:02d}.npz");info=json.loads((runfolder/f"s{sid:02d}.json").read_text());b=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz");static=np.load(ROOT/"phase53"/("static_"+variant)/f"s{sid:02d}.npz")
   y=z["y"];p=z["p"];assert np.array_equal(y,b["y"]) and np.array_equal(y,static["y"])
   assert info["nonwritable_unchanged"] and info["radius_verified"] and np.isfinite(p).all()
   assert abs(metrics(y,p)["macro_f1"]-info["metrics"]["macro_f1"])<1e-12
   newonline+=int(seed!=17)
   for smooth in [1.,.8,.5]:
    preds=[p,b["p"],static["predictions"]];score=[100*metrics(y,q if smooth==1 else ema(q,smooth))["macro_f1"] for q in preds]
    online_rows.append(dict(seed=seed,variant=variant,subject=sid,lr=lr,ema=smooth,macro_f1=score[0],frozen_f1=score[1],static_f1=score[2],gain_vs_frozen_pp=score[0]-score[1],gain_vs_static_pp=score[0]-score[2]))
pulses=pd.DataFrame(pulse_rows);online=pd.DataFrame(online_rows);valid=pd.concat(validity,ignore_index=True)
pulses.to_csv(o/"pulse_people.csv",index=False);online.to_csv(o/"online_people.csv",index=False);valid.to_csv(o/"reconstruction_validity.csv",index=False)
ps=pulses.groupby(["seed","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy_gain_pp=("accuracy_gain_pp","mean"),sample_f1_gain_pp=("sample_f1_gain_pp","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),positive_people=("future_ce_gain",lambda x:int((x>0).sum())))
ps.to_csv(o/"pulse_seed_summary.csv")
ss=online.groupby(["seed","variant","ema"])[["macro_f1","frozen_f1","static_f1","gain_vs_frozen_pp","gain_vs_static_pp"]].mean();ss.to_csv(o/"online_seed_summary.csv")
within=online.groupby(["subject","variant","ema"])[["macro_f1","frozen_f1","static_f1","gain_vs_frozen_pp","gain_vs_static_pp"]].mean()
within.to_csv(o/"online_seedmean_people.csv")
means=within.groupby(["variant","ema"]).mean();means.to_csv(o/"online_seedmean_summary.csv")
audit=dict(status="PASS",pulse_rows_including_reused=totalp,new_pulse_rows=totalp-512,new_online_trajectories=newonline,all_online_trajectories=24,graph_eager_new_cases=12,maximum_ce_recompute_error=max_error,label_indices_exactly_match_seed17=True,classifier_state_exact_unchanged=True,auxiliary_only_seeds=[17,29,43],base_classifier_seed=17,source_only=True,no_prediction_ensemble=True,actual_epoch_orders_verified=True,invalid_seed_reset_attempt_excluded=True,no_automatic_target_advance=True)
(o/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>固定设置的辅助头训练种子重复</h1><p>基础分类器seed17和频谱读出初始化固定。仅辅助头训练随机种子17/29/43，17复用；每种目标同样317/1268标签、五轮、最终轮，无超参数或轮次重选。在线使用阶段55预先固定的当前头1e-4、未来头0.01，持续更新与共同EMA网格。此试验衡量辅助头训练顺序的敏感性，不能称为三个独立分类器的稳定性或新盲测。</p><p>首轮种子被模型构造器覆盖，已归档为无效重复。修正为独立CPU排列生成器后重跑并逐轮核对317个样本的排列；历史seed17保持原有结果，其排列来自模型初始化后的全局RNG状态。<a href="seed_reset_incident.json">问题与证据</a>。</p><h2>单步新鲜写入：持出四人</h2>'+ps.to_html(float_format=lambda x:f"{x:.7f}")+'<h2>完整源验证：每种子</h2>'+ss.to_html(float_format=lambda x:f"{x:.5f}")+'<h2>先在每人平均三个种子指标，再平均人</h2>'+means.to_html(float_format=lambda x:f"{x:.5f}")+'<p>均值不是预测集成。同一个人的三个种子不算三个独立受试者。当前头仅与317标签静态校准、未来头仅与1268标签静态校准比较。源四人已多轮使用，均为探索性验证；无自动目标人群晋级。</p><h2>重建有效性</h2>'+valid.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p><a href="protocol.json">固定协议</a> · <a href="audit.json">独立审计</a> · <a href="training_code_provenance.json">训练代码差异范围</a></p></html>'
(o/"report.html").write_text(html)
print(ps.to_string());print(ss.to_string());print(means.to_string());print(audit)

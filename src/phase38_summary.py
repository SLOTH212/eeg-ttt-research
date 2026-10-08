from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
o=ROOT/"phase38";rows=[];aux=[];count=0;source_count=0
for seed in [17,29,43]:
 folder=ROOT/"phase35/young" if seed==17 else o/f"young_seed{seed}";parent=ROOT/"phase28/young_joint_0" if seed==17 else ROOT/"phase33"/f"young_joint_0_seed{seed}"
 if not (folder/"complete.json").exists():continue
 meta=json.loads((folder/"complete.json").read_text());a=torch.load(folder/"model.pt",map_location="cpu",weights_only=False)["state"];b=torch.load(parent/"model.pt",map_location="cpu",weights_only=False)["state"];allowed={"mask_token","aux_head.weight","aux_head.bias"};assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a if k not in allowed);assert set(meta["trainable_names"])==allowed;assert hashlib.sha256((parent/"model.pt").read_bytes()).hexdigest()==meta["parent_classifier_sha256"];assert abs(meta["best_aux_ce"]-min([np.mean([v["masked_ce"] for v in meta["initial_auxiliary"].values()])]+[h["auxiliary_ce"] for h in meta["history"]]))<1e-10
 for sid in [5,15,25,35]:
  p=np.load(folder/f"s{sid:02d}.npz");q=np.load(parent/f"s{sid:02d}.npz");assert np.array_equal(p["y"],q["y"]) and np.array_equal(p["p"],q["p"]);source_count+=1
 aux.append(dict(classifier_seed=seed,auxiliary_seed=17,epoch=meta["best_epoch"],auxiliary_ce=meta["best_aux_ce"],shuffled_ce=np.mean([v["input_shuffled_ce"] for v in meta["validation"]]),classifier_unchanged_exact=True))
 for variant in ["fitted","unfitted"]:
  case=ROOT/"phase37"/f"young_{variant}_0.01" if seed==17 else o/"evaluation"/f"young_{variant}_0.01_seed{seed}"
  if not (case/"complete.json").exists():continue
  lock=json.loads((case/"selection.json").read_text());checkpoint=folder/"model.pt" if variant=="fitted" else parent/"model.pt";assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==lock["checkpoint_sha256"];assert lock["lr"]==.01
  for sid in [5,15,25,35]:
   z=np.load(case/f"s{sid:02d}.npz");base=np.load(parent/f"s{sid:02d}.npz");rec=json.loads((case/f"s{sid:02d}.json").read_text());root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];assert np.array_equal(y,z["y"]) and np.array_equal(y,base["y"]);assert rec["nonwritable_unchanged"] and rec["radius_verified"];streams={"metrics":z["p"],"adapted_ema":ema(z["p"],.8),"frozen":base["p"],"weak_ema":ema(base["p"],.8),"strong_ema":ema(base["p"],.5)}
   for method,p in streams.items():
    assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;mm=metrics(y,p);assert abs(mm["macro_f1"]-rec[method]["macro_f1"])<1e-12;rows.append(dict(classifier_seed=seed,variant=variant,subject=sid,method=method,f1=100*mm["macro_f1"],ce=mm["ce"]))
   count+=1
d=pd.DataFrame(rows);d.to_csv(o/"participants.csv",index=False);pd.DataFrame(aux).to_csv(o/"auxiliary_summary.csv",index=False);complete=count==24 and source_count==12;means=d.groupby(["classifier_seed","variant","method"])[["f1","ce"]].mean().reset_index();means.to_csv(o/"seed_means.csv",index=False);contrasts=[]
for seed,g in d.groupby("classifier_seed"):
 for method,ref,label in [("metrics","frozen","fitted_online_vs_frozen"),("adapted_ema","weak_ema","fitted_onlineEMA_vs_sameEMA"),("metrics","strong_ema","fitted_online_vs_strongEMA"),("adapted_ema","strong_ema","fitted_onlineEMA_vs_strongEMA")]:
  h=g[g.variant=="fitted"].pivot(index="subject",columns="method",values="f1")
  if method in h and ref in h:
   delta=h[method]-h[ref];contrasts.append(dict(classifier_seed=seed,comparison=label,delta_pp=delta.mean(),positive_people=int((delta>0).sum()),worst_person_pp=delta.min()))
 for method in ["metrics","adapted_ema"]:
  h=g[g.method==method].pivot(index="subject",columns="variant",values="f1")
  if "fitted" in h and "unfitted" in h:
   delta=h.fitted-h.unfitted;contrasts.append(dict(classifier_seed=seed,comparison="fitted_vs_unfitted_"+method,delta_pp=delta.mean(),positive_people=int((delta>0).sum()),worst_person_pp=delta.min()))
c=pd.DataFrame(contrasts);c.to_csv(o/"contrasts.csv",index=False);(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=complete,exact_unchanged_classifier_streams=source_count,adapted_streams=count,recomputed_method_streams=count*5,parent_checkpoint_hashes_verified=True,selected_auxiliary_epoch_verified=True,raw_labels_verified=True,metrics_recomputed=True),indent=2))
style='<style>body{font:15px/1.65 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ccc}section{overflow:auto}</style>'
html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>辅助头单独训练：固定设置的三分类器种子源复现</h1><p>分类器种子17/29/43，均为已有零辅助权重匹配模型。辅助训练随机种子固定17，额外10轮、仅按辅助交叉熵选择轮次。分类器参数与源验证概率必须保持完全相同。固定连续掩码更新LR0.01；不追加搜索步长。</p><p>这是阶段37年轻来源的探索性复现。阶段37原始在线分数未通过推进条件；在线更新再加相同弱EMA的观察促使此项固定设置检查，不能改写原判定。四名源验证人群重复使用，没有新目标人群或盲测。</p>'+f'<p>已完成{source_count}/12条原始分类流一致性核对、{count}/24条在线轨迹审计，每条复算5种方法。</p><h2>辅助任务及分类器不变性</h2><section>'+pd.DataFrame(aux).to_html(index=False,float_format=lambda x:f"{x:.5f}")+'</section><h2>各训练种子的配对差异</h2><section>'+c.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'</section>'
if complete:
 person=d.groupby(["variant","subject","method"])[["f1","ce"]].mean().reset_index();person.to_csv(o/"mean3_person_metrics.csv",index=False);avg=c.groupby("comparison").delta_pp.mean().reset_index();avg.to_csv(o/"mean3_contrasts.csv",index=False);html+='<h2>三种子平均差异</h2><p>先在同一受试者内平均种子各自性能，不是预测集成；只有四人，不提供具有确认性含义的置信区间。个体差异保留在CSV中。</p><section>'+avg.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'</section>'
html+='<p><a href="participants.csv">逐人逐种子结果</a> · <a href="contrasts.csv">全部比较</a> · <a href="audit.json">独立审计</a> · <a href="protocol.json">固定设置协议</a></p></html>';(o/"report.html").write_text(html);print("COMPLETE",complete,"SOURCE",source_count,"ONLINE",count);print(c.to_string(index=False))

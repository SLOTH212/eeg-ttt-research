from common import *
from sklearn.metrics import f1_score
import pandas as pd,hashlib
o=ROOT/"phase73";pr=json.loads((o/"protocol.json").read_text());assert (o/"complete.json").exists();cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);records=[];maxerr=0.;maxnormerr=0.;count=0
chosen=pr["checkpoint"];assert hashlib.sha256((Path(chosen["folder"])/"model.pt").read_bytes()).hexdigest()==chosen["checkpoint_sha256"]
for sid in pr["controllers"]:
 data=json.loads((o/f"s{sid:02d}.json").read_text());assert data["state_restored_exact"];root=ROOT/("data/processed" if sid<20 else "phase2/processed");truth=np.load(root/f"s{sid:02d}_meta.npz")["y"];times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"]
 lookup={(r["t"],r["lr"],r["variant"]):r for r in data["rows"]};assert len(lookup)==len(times)*9
 for lr in pr["learning_rates"]:
  for variant in pr["variants"]:
   rows=[r for r in data["rows"] if r["lr"]==lr and r["variant"]==variant];assert [r["t"] for r in rows]==times.tolist();g=[];b=[];corr=0;harm=0;changes=0;yy=[];yp=[];wp=[]
   for rr in rows:
    y=np.array(rr["future_labels"]);base=np.array(rr["future_base_probabilities"]);write=np.array(rr["future_write_probabilities"]);assert np.array_equal(y,truth[rr["t"]+1:rr["t"]+5]);assert (base>0).all() and (write>0).all() and np.isfinite(write).all();assert abs(write.sum(1)-1).max()<1e-12
    delta=np.log(write[np.arange(4),y])-np.log(base[np.arange(4),y]);gain=float(delta.mean());balanced=float((delta*cw[y]).mean());err=max(abs(gain-rr["future_ce_gain"]),abs(balanced-rr["balanced_ce_gain"]));maxerr=max(maxerr,err);assert err<1e-10
    before=base.argmax(1);after=write.argmax(1);c=int(((before!=y)&(after==y)).sum());h=int(((before==y)&(after!=y)).sum());change=int((before!=after).sum());assert c==rr["corrected"] and h==rr["harmed"] and change==rr["argmax_changes"];assert rr["actual_update_norm"]<=lr+1e-5
    ref=lookup[(rr["t"],lr,"refitted")];assert rr["future_base_probabilities"]==ref["future_base_probabilities"] and rr["masked_patches"]==ref["masked_patches"]
    if variant=="original_normmatched":
     normerr=abs(rr["actual_update_norm"]-ref["actual_update_norm"]);maxnormerr=max(maxnormerr,normerr);assert normerr<2e-6
    corr+=c;harm+=h;changes+=change;g.append(gain);b.append(balanced);yy.extend(y);yp.extend(before);wp.extend(after);count+=1
   records.append(dict(variant=variant,subject=sid,lr=lr,pulses=len(rows),future_ce_gain=np.mean(g),balanced_ce_gain=np.mean(b),accuracy_gain_pp=100*(corr-harm)/len(yy),corrected=corr,harmed=harm,changed=changes,aux_decrease_fraction=np.mean([r["aux_loss_after"]<r["aux_loss_before"] for r in rows]),mean_update_norm=np.mean([r["actual_update_norm"] for r in rows]),sample_macro_f1_gain_pp=100*(f1_score(yy,wp,labels=range(5),average="macro",zero_division=0)-f1_score(yy,yp,labels=range(5),average="macro",zero_division=0))))
people=pd.DataFrame(records);people.to_csv(o/"people.csv",index=False)
summary=people.groupby(["variant","lr"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy_gain_pp=("accuracy_gain_pp","mean"),sample_macro_f1_gain_pp=("sample_macro_f1_gain_pp","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),changed=("changed","sum"),mean_update_norm=("mean_update_norm","mean"),aux_decrease_fraction=("aux_decrease_fraction","mean"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())))
summary.to_csv(o/"summary.csv");decisions=[]
for (variant,lr),r in summary.iterrows():
 passed=r.future_ce_gain>0 and r.positive_people>=3 and r.balanced_ce_gain>=0 and r.accuracy_gain_pp>=0
 decisions.append(dict(variant=variant,lr=lr,primary=variant=="refitted" and lr==.01,future_ce_gain=r.future_ce_gain,balanced_ce_gain=r.balanced_ce_gain,accuracy_gain_pp=r.accuracy_gain_pp,positive_people=int(r.positive_people),passes_fresh_write_screen=bool(passed)))
(o/"decisions.json").write_text(json.dumps(decisions,indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",rows=count,people=4,writable_parameters=9600,labels_and_all_probabilities_verified=True,maximum_ce_error=maxerr,maximum_matched_update_norm_error=maxnormerr,source_checkpoint_hash_verified=True,state_restoration_verified=True,no_online_or_target_claim=True),indent=2))
(o/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>CBraMod 波形重建实际单步写入</h1><p>固定分类器种子17。四名源控制器各64时点，当前窗预测后仅更新9600个 LayerNorm 参数，观测随后四窗；未来暂停其他更新。主条件为重新拟合投影、学习率0.01。原始头只作诊断对照，不作为通过有效性的部署候选。额外等更新范数对照用于区分方向和幅度。</p>'+summary.to_html()+'<p>判据：平均 CE 收益为正，至少3/4人正，平衡 CE 与正确数非负。不能将这些单次收益相加为在线 Macro-F1。原始头重建失败和三种子静态结果保持原结论。</p>'+pd.DataFrame(decisions).to_html(index=False)+'<p><a href="audit.json">独立审计</a> · <a href="protocol.json">固定协议</a></p>')
print(summary.to_string());print(decisions)

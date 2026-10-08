from common import *
from sklearn.metrics import f1_score
import pandas as pd,hashlib
o=ROOT/"phase69";pr=json.loads((o/"protocol.json").read_text());assert (o/"execution_complete.json").exists();cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);records=[];maxerr=0.;count=0
for kind in ["pretrained","random"]:
 writable=json.loads((o/kind/"writable.json").read_text());assert writable["parameters"]==9600 and len(writable["names"])==48 and all(".norm1." in n or ".norm2." in n for n in writable["names"])
 chosen=json.loads((ROOT/"phase66/selection.json").read_text())["selected"][kind];assert hashlib.sha256((Path(chosen["folder"])/"model.pt").read_bytes()).hexdigest()==chosen["checkpoint_sha256"]
 for sid in pr["controllers"]:
  data=json.loads((o/kind/f"s{sid:02d}.json").read_text());assert data["state_restored_exact"];root=ROOT/("data/processed" if sid<20 else "phase2/processed");truth=np.load(root/f"s{sid:02d}_meta.npz")["y"];times=np.load(ROOT/"phase48/masked"/f"s{sid:02d}_gradients.npz")["times"]
  for lr in pr["learning_rates"]:
   rows=[r for r in data["rows"] if r["lr"]==lr];assert [r["t"] for r in rows]==times.tolist();g=[];b=[];corr=0;harm=0;changes=0;yy=[];yp=[];wp=[]
   for rr in rows:
    y=np.array(rr["future_labels"]);base=np.array(rr["future_base_probabilities"]);write=np.array(rr["future_write_probabilities"]);assert np.array_equal(y,truth[rr["t"]+1:rr["t"]+5]);assert (base>0).all() and (write>0).all() and np.isfinite(write).all();assert abs(write.sum(1)-1).max()<1e-12
    delta=np.log(write[np.arange(4),y])-np.log(base[np.arange(4),y]);gain=float(delta.mean());balanced=float((delta*cw[y]).mean());err=max(abs(gain-rr["future_ce_gain"]),abs(balanced-rr["balanced_ce_gain"]));maxerr=max(maxerr,err);assert err<1e-10
    before=base.argmax(1);after=write.argmax(1);c=int(((before!=y)&(after==y)).sum());h=int(((before==y)&(after!=y)).sum());change=int((before!=after).sum());assert c==rr["corrected"] and h==rr["harmed"] and change==rr["argmax_changes"];assert rr["actual_update_norm"]<=lr+1e-5
    corr+=c;harm+=h;changes+=change;g.append(gain);b.append(balanced);yy.extend(y);yp.extend(before);wp.extend(after);count+=1
   records.append(dict(kind=kind,subject=sid,lr=lr,pulses=len(rows),future_ce_gain=np.mean(g),balanced_ce_gain=np.mean(b),accuracy_gain_pp=100*(corr-harm)/len(yy),corrected=corr,harmed=harm,changed=changes,sample_macro_f1_gain_pp=100*(f1_score(yy,wp,labels=range(5),average="macro",zero_division=0)-f1_score(yy,yp,labels=range(5),average="macro",zero_division=0))))
people=pd.DataFrame(records);people.to_csv(o/"people.csv",index=False)
summary=people.groupby(["kind","lr"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),accuracy_gain_pp=("accuracy_gain_pp","mean"),sample_macro_f1_gain_pp=("sample_macro_f1_gain_pp","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),changed=("changed","sum"),positive_people=("future_ce_gain",lambda v:int((v>0).sum())))
summary.to_csv(o/"summary.csv");decisions=[]
for (kind,lr),row in summary.iterrows():
 passed=row.future_ce_gain>0 and row.positive_people>=3 and row.balanced_ce_gain>=0 and row.accuracy_gain_pp>=0
 decisions.append(dict(kind=kind,lr=lr,primary=kind=="pretrained" and lr==pr["primary_lr"],future_ce_gain=row.future_ce_gain,balanced_ce_gain=row.balanced_ce_gain,accuracy_gain_pp=row.accuracy_gain_pp,positive_people=int(row.positive_people),passes_fresh_write_screen=bool(passed)))
(o/"decisions.json").write_text(json.dumps(decisions,indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",rows=count,carriers=2,people=4,writable_parameters=9600,labels_and_all_probabilities_verified=True,maximum_ce_error=maxerr,source_checkpoint_hashes_verified=True,base_state_restored_aftereachpulse=True,actual_update_norm_bounded=True,no_online_or_target_claim=True),indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>CBraMod标准熵更新的实际新鲜写入</h1><p>分类载体和检查点仅按阶段66来源原始F1选定。四名互斥控制器受试者，每人64时点，比较一步更新后未来四窗与源状态；标准预测熵，9600个Transformer LayerNorm仿射参数可写，卷积GroupNorm及分类头固定。三个步长均预先列出，预训练0.001为主条件。原波形重建头未通过有效性检查，本阶段不运行该目标。</p>'+summary.to_html(float_format=lambda x:f"{x:.8f}")+'<h2>预定单步条件</h2>'+pd.DataFrame(decisions).to_html(index=False,float_format=lambda x:f"{x:.8f}")+'<p>要求CE正、至少3/4人正效用、平衡CE与正确数非负。次级步长不得替换主条件结果；即使通过，仍需实际连续源侧轨迹及强冻结/EMA对照。此处暂停未来其他更新，不能将单步收益相加视为在线成绩。所有人此前已在项目中出现；随机和预训练载体分类基线不同，效用比较有条件。</p><p><a href="protocol.json">协议</a> · <a href="audit.json">独立复算</a> · <a href="decisions.json">判定</a></p></html>';(o/"report.html").write_text(html);print(summary.to_string());print(decisions)

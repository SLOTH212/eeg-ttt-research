from common import *
import pandas as pd,hashlib
O=ROOT/"phase87";assert (O/"complete.json").exists();assert json.loads((O/"execution_audit.json").read_text())["status"]=="PASS";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((ROOT/"phase86/protocol.json").read_bytes()).hexdigest()==pr["parent_protocol_sha256"]
for name,sha in pr["models"].items():assert hashlib.sha256((ROOT/"phase86"/name/"head.pt").read_bytes()).hexdigest()==sha
cw=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);rows=[];ceerr=0.
for sid in pr["controllers"]+pr["validation"]:
 d=json.loads((O/f"s{sid:02d}.json").read_text());assert d["state_restored_exact"] and len(d["rows"])==320
 for r in d["rows"]:
  y=np.array(r["future_labels"]);b=np.array(r["future_base_probabilities"]);w=np.array(r["future_write_probabilities"]);assert (b>0).all() and (w>0).all();gain=np.log(w[np.arange(4),y])-np.log(b[np.arange(4),y]);err=abs(gain.mean()-r["future_ce_gain"]);ceerr=max(ceerr,err);assert err<1e-12 and abs((gain*cw[y]).mean()-r["balanced_ce_gain"])<1e-12
  bp=b.argmax(1);wp=w.argmax(1);assert int(((bp!=y)&(wp==y)).sum())==r["corrected"];assert int(((bp==y)&(wp!=y)).sum())==r["harmed"];assert int((bp!=wp).sum())==r["argmax_changes"]
  rows.append({k:v for k,v in r.items() if k not in ["future_labels","future_base_probabilities","future_write_probabilities"]})
df=pd.DataFrame(rows);assert len(df)==2560
pv=df.pivot(index=["group","subject","t"],columns="variant",values="actual_update_norm");normerr=float(abs(pv.future-pv.initial_normmatched_future).max());assert normerr<1e-6
df.to_csv(O/"pulses.csv",index=False);person=df.groupby(["group","subject","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),current_ce_gain=("current_ce_gain","mean"),aux_loss_before=("aux_loss_before","mean"),aux_loss_after=("aux_loss_after","mean"),actual_update_norm=("actual_update_norm","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),argmax_changes=("argmax_changes","sum")).reset_index();person.to_csv(O/"people.csv",index=False)
summary=person.groupby(["group","variant"]).agg(future_ce_gain=("future_ce_gain","mean"),balanced_ce_gain=("balanced_ce_gain","mean"),current_ce_gain=("current_ce_gain","mean"),positive_people=("future_ce_gain",lambda x:int((x>0).sum())),corrected=("corrected","sum"),harmed=("harmed","sum"),argmax_changes=("argmax_changes","sum"),update_norm=("actual_update_norm","mean"));summary.to_csv(O/"summary.csv")
valid=pd.read_csv(O/"reconstruction_validation.csv");mseerr=0.
for r in valid.itertuples():
 a=np.load(O/"validation_cases"/f"{r.variant}_s{r.subject:02d}.npz");counts=a["mask"].sum((1,2));start=0;values=[]
 for count in counts:
  sl=slice(start,start+count);target=a["target"][sl].astype(float);pred=a["prediction"][sl].astype(float);sh=a["shuffle_prediction"][sl].astype(float);values.append([np.square(pred-target).mean(),np.square(target).mean(),np.square(sh-target).mean()]);start+=count
 values=np.array(values);err=float(abs(values-a["window_metrics"]).max());mseerr=max(mseerr,err);assert err<1e-12 and abs(values[:,0].mean()-r.mse)<1e-12
training={}
for which in ["future","current","reconstruction"]:
 d=json.loads((ROOT/"phase86"/which/"complete.json").read_text());assert d["classifier_unchanged_exact"];training[which]={k:d[k] for k in ["pulses","epochs","label_count","seconds"]}
checks={}
for group in ["controllers","validation"]:
 s=summary.loc[group];p=s.loc["future"];checks[group]=dict(positive_future_ce=bool(p.future_ce_gain>0),positive_at_least3of4=bool(p.positive_people>=3),balanced_ce_nonnegative=bool(p.balanced_ce_gain>=0),corrected_at_least_harmed=bool(p.corrected>=p.harmed),beats_initial=bool(p.future_ce_gain>s.loc["initial"].future_ce_gain),beats_current=bool(p.future_ce_gain>s.loc["current"].future_ce_gain),beats_reconstruction=bool(p.future_ce_gain>s.loc["reconstruction"].future_ce_gain),beats_normmatched_initial=bool(p.future_ce_gain>s.loc["initial_normmatched_future"].future_ce_gain))
checks["reconstruction_all4"]=bool(valid[valid.variant=="future"].validity_pass.all());advance=all(checks["controllers"].values()) and all(checks["validation"][k] for k in ["positive_future_ce","positive_at_least3of4","balanced_ce_nonnegative","corrected_at_least_harmed"]) and checks["reconstruction_all4"]
decision=dict(advance=advance,checks=checks,interpretation="All requirements conjunctive. Failed reconstruction cannot be hidden by task CE gain; no online/target rollout unless all gates pass.",training_cost=training);(O/"decision.json").write_text(json.dumps(decision,indent=2))
audit=dict(status="PASS",pulse_rows=2560,people=8,max_ce_error=ceerr,max_normmatching_error=normerr,max_independent_reconstruction_mse_error=mseerr,reconstruction_cases=16,protocol_and_head_hashes_verified=True,no_probability_floor=True,classification_parameters_unchanged=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>CBraMod辅助波形头元训练：实际写入与重建双重检验</h1><p>固定原分类器，初始辅助头取已验证的phase72重拟合投影；额外训练仅改变40,200个辅助头参数。内层只对9,600个LayerNorm仿射参数做当前无标签重建更新，步长0.01、范数裁剪1；当前预测先于写入，评分只看随后4个片段。所有脉冲恢复初始分类器，属于单次写入能力检验，不能等同持续在线收益。</p><p>5位额外源训练者，共317脉冲、固定5轮。后续目标使用1,268个分期标签，当前目标317个，仅重建目标0个；绝非零成本复用预训练任务。两组各4位评估受试者均未参与本轮元训练，但此前项目已观察，非盲测。</p>'+summary.reset_index().to_html(index=False,float_format=lambda x:f"{x:.8f}")+'<h2>重建有效性</h2>'+valid.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>固定推进门槛</h2><pre>'+json.dumps(decision,ensure_ascii=False,indent=2)+'</pre><p>后续目标必须同时胜过KEEP、初始头、当前目标、仅重建继续训练及初始方向的等范数写入，且收益跨人复现、类别加权收益与正确性非负、重建仍有效。不能只挑某个CE数字报告成功。</p><p><a href="audit.json">独立结果审计</a> · <a href="../phase86/implementation_audit.json">二阶梯度审计</a> · <a href="../phase86/protocol.json">固定元训练方案</a> · <a href="people.csv">逐人结果</a></p>')
print(summary.to_string());print(valid.to_string(index=False));print(decision);print(audit)

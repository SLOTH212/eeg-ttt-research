from phase28_joint_train import JointCarrier
from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
o=ROOT/"phase45";protocol=json.loads((o/"protocol.json").read_text());static=o/"mixed_static_matched";meta=json.loads((static/"complete.json").read_text());assert meta["label_count"]==4904;expected={}
for sid in protocol["controllers"]:
 idx=np.unique(np.concatenate([np.load(ROOT/"phase43"/f"lr0.01_{t}_s{sid:02d}/pulses.npz")["future_labels_used"] for t in ["continuous","reset64"]]));expected[str(sid)]=idx.tolist()
assert expected==meta["labels_used"];assert meta["best_val_f1"]==max([meta["initial_val_f1"]]+[h["val_f1"] for h in meta["history"]]);ck=torch.load(static/"model.pt",map_location="cpu",weights_only=False);parent=torch.load(ROOT/"phase35/mixed/model.pt",map_location="cpu",weights_only=False)["state"];assert ck["labels_used"]==expected;model=JointCarrier("mixed");allowed=set()
for name,module in model.named_modules():
 if isinstance(module,nn.LayerNorm):allowed.update(name+"."+n for n,p in module.named_parameters(recurse=False))
assert sum(parent[k].numel() for k in allowed)==10960;assert all(torch.equal(parent[k],ck["state"][k]) for k in parent if k not in allowed)
for k in allowed:
 v=parent[k];radius=.1*max(float(v.square().mean().sqrt()),1)*v.numel()**.5;assert float((ck["state"][k]-v).norm())<=radius+1e-5
del model,ck,parent
assert json.loads((o/"closed_loop_audit.json").read_text())["status"]=="PASS";rows=[];cases=0
def add(sid,name,y,p,write=0.,reset=0.):
 assert np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5;mm=metrics(y,p);rows.append(dict(subject=sid,method=name,f1=100*mm["macro_f1"],ce=mm["ce"],write_fraction=write,reset_fraction=reset));return mm
for sid in protocol["subjects"]:
 root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"];base=np.load(ROOT/"phase35/mixed"/f"s{sid:02d}.npz");assert np.array_equal(y,base["y"]);add(sid,"frozen",y,base["p"]);add(sid,"weak_ema",y,ema(base["p"],.8));add(sid,"strong_ema",y,ema(base["p"],.5));z=np.load(ROOT/"phase37/mixed_fitted_0.01"/f"s{sid:02d}.npz");assert np.array_equal(y,z["y"]);add(sid,"WRITE",y,z["p"],1.);add(sid,"WRITE_ema08",y,ema(z["p"],.8),1.);z=np.load(static/f"s{sid:02d}.npz");assert np.array_equal(y,z["y"]);mm=add(sid,"static_matched",y,z["predictions"]);ref=next(v for v in meta["results"] if v["subject"]==sid);assert abs(mm["macro_f1"]-ref["metrics"]["macro_f1"])<1e-12;add(sid,"static_matched_ema08",y,ema(z["predictions"],.8));add(sid,"static_matched_ema05",y,ema(z["predictions"],.5))
 for policy in protocol["new_policies"]:
  folder=o/policy;done=folder/f"s{sid:02d}.json"
  if not done.exists():continue
  rec=json.loads(done.read_text());z=np.load(folder/f"s{sid:02d}.npz");assert np.array_equal(y,z["y"]);assert rec["nonwritable_unchanged"] and rec["radius_verified"];action=z["actions"];assert len(action)==len(y)
  if policy=="anchor_guard":
   expect=np.where(abs(z["p"]-base["p"]).sum(1)>.05,1,2);assert np.array_equal(action,expect)
  else:assert np.isin(action,[0,2]).all()
  if policy.startswith("cadence"):assert np.array_equal(action,np.where(np.arange(len(y))%int(policy.removeprefix("cadence"))==0,2,0))
  if policy=="low_entropy_write":
   ent=-(z["p"]*np.log(np.clip(z["p"],1e-12,1))).sum(1);assert np.array_equal(action,np.where(ent<.5,2,0))
  write=float((action==2).mean());reset=float((action==1).mean());assert write==rec["write_fraction"] and reset==rec["reset_fraction"];mm=add(sid,policy,y,z["p"],write,reset);assert abs(mm["macro_f1"]-rec["metrics"]["macro_f1"])<1e-12;mm=add(sid,policy+"_ema08",y,ema(z["p"],.8),write,reset);assert abs(mm["macro_f1"]-rec["adapted_ema"]["macro_f1"])<1e-12;cases+=1
df=pd.DataFrame(rows);df.to_csv(o/"participants.csv",index=False);sizes=df.groupby("method").subject.nunique();valid=sizes[sizes==4].index;means=df[df.method.isin(valid)].groupby("method").mean(numeric_only=True).drop(columns="subject").reset_index();means.to_csv(o/"summary.csv",index=False);complete=cases==24;contrasts=[];q=df.pivot(index="subject",columns="method",values="f1")
if "future36" in valid:
 for control in [k for k in ["current36","WRITE","low_entropy_write","anchor_guard","cadence2","cadence4","frozen","weak_ema","strong_ema","static_matched","static_matched_ema08","static_matched_ema05"] if k in valid]:
  delta=q.future36-q[control];contrasts.append(dict(comparison="future36_vs_"+control,delta_pp=float(delta.mean()),positive_people=int((delta>0).sum()),worst_person_pp=float(delta.min())))
 for control in [k+"_ema08" for k in ["current36","WRITE","low_entropy_write","anchor_guard","cadence2","cadence4","static_matched"] if k+"_ema08" in valid]+["weak_ema"]:
  delta=q.future36_ema08-q[control];contrasts.append(dict(comparison="future36EMA_vs_"+control,delta_pp=float(delta.mean()),positive_people=int((delta>0).sum()),worst_person_pp=float(delta.min())))
pd.DataFrame(contrasts).to_csv(o/"contrasts.csv",index=False);decision=dict(complete=complete,criterion=protocol["advance"],advance=False)
if complete:
 controls=["current36","WRITE","low_entropy_write","anchor_guard","cadence2","cadence4","frozen","weak_ema","strong_ema","static_matched"];best=max(float(q[k].mean()) for k in controls);decision.update(future_f1=float(q.future36.mean()),best_control_f1=best,advance=bool(q.future36.mean()>best),best_control=max(controls,key=lambda k:q[k].mean()),primary_control_set=controls,best_reported_method=means.loc[means.f1.idxmax(),"method"],best_reported_f1=float(means.f1.max()))
(o/"advance_decision.json").write_text(json.dumps(decision,indent=2));(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=complete,closed_loop_policy_cases=cases,recomputed_method_streams=len(rows),static_future_label_count=4904,static_future_label_indices_exact=True,static_epoch_choice_verified=True,static_nonwritable_parameters_exact=True,static_radius_verified=True,fixed_policy_actions_recomputed=True,learned_policy_future_prefix_audit=True,raw_labels_verified=True,metrics_recomputed=True),indent=2));style='<style>body{font:15px/1.65 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}section{overflow:auto}</style>'
html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>九人收益模型的真实闭环源验证</h1><p>阶段43只有36维普通未来收益、限保留/写入的候选达到预设条件；以其全九人拟合模型、固定LR0.01进入四人源验证。对照包括同容量当前收益、固定低熵写入、源预测偏离重置、每2/4窗口写入、连续写入及冻结/平滑。</p><p>静态校准获得与未来收益训练完全相同的4904个唯一未来标签，只训练同10960个LayerNorm参数、同源半径，20轮中含原始epoch0按源验证选择；这匹配标签和可更新参数范围，不是相同计算预算。当前收益控制器使用较少的当前标签，不能把它当作标签量完全相同的对照。</p>'+f'<p>静态校准选择第{meta["best_epoch"]}轮；标签索引、参数不变性和半径独立核对。真实策略当前完成{cases}/24条轨迹、共{len(rows)}条方法指标复算。未来/当前两种策略的前32窗口预测、动作及参数状态在扰动未来输入后完全一致，且审计前缀包含保留与写入两种动作。</p><section>'+means.to_html(index=False,float_format=lambda v:f"{v:.5f}")+'</section><h2>未来收益策略的配对差异</h2><section>'+pd.DataFrame(contrasts).to_html(index=False,float_format=lambda v:f"{v:.5f}")+'</section>'+f'<p>完整后是否推进目标策略测试：{decision["advance"] if complete else "等待全部对照"}。四名源验证人群反复使用，不能称独立确认；未通过则不进入目标策略评估。</p><p><a href="participants.csv">逐人分数</a> · <a href="audit.json">独立审计</a> · <a href="closed_loop_audit.json">闭环因果审计</a> · <a href="advance_decision.json">预设推进判定</a></p></html>';(o/"report.html").write_text(html);print(means.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False));print("DECISION",decision)

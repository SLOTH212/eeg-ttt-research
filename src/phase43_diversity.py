from common import *
import pandas as pd,pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
o=ROOT/"phase43";protocol=json.loads((o/"protocol.json").read_text());assert json.loads((o/"pulses_complete.json").read_text())["complete"];assert json.loads((o/"feature_causality_audit.json").read_text())["status"]=="PASS";weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"]);parts=[];maxerr=0.
for path in o.glob("lr*/complete.json"):
 meta=json.loads(path.read_text());z=np.load(path.parent/"pulses.npz");x=z["features"];yy=z["future_labels"];idx=np.indices(yy.shape);a=z["future_keep"];alog=np.log(np.clip(a[idx[0],idx[1],yy],1e-38,1));correct=a.argmax(-1)==yy;balanced=[];acc=[];bacc=[]
 for action,col in [("reset",0),("commit",1)]:
  b=z["future_"+action];blog=np.log(np.clip(b[idx[0],idx[1],yy],1e-38,1));gain=blog-alog;err=float(abs(gain.mean(1)-z["future_gain"][:,col]).max());assert err<2e-5;maxerr=max(maxerr,err);balanced.append((gain*weights[yy]).mean(1));difference=(b.argmax(-1)==yy).astype(float)-correct;acc.append(difference.mean(1));bacc.append((difference*weights[yy]).mean(1))
 sid=meta["subject"];raw=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_meta.npz")["y"];assert np.array_equal(yy,np.stack([raw[t+1:t+5] for t in z["t"]]));parts.append(dict(features=x,subjects=np.full(len(x),sid),times=z["t"],future_labels=yy,ordinary_future=z["future_gain"],ordinary_current=z["current_gain"],balanced_future=np.column_stack(balanced),balanced_current=z["current_gain"]*weights[z["current_class"]][:,None],accuracy=np.column_stack(acc),balanced_accuracy=np.column_stack(bacc)))
assert len(parts)==18;d={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};assert all(np.isfinite(v).all() for v in d.values());x=d["features"];ss=d["subjects"];tt=d["times"];original=np.isin(ss,protocol["original_controllers"]);outcomes={k:np.column_stack([np.zeros(len(x)),d[k]]) for k in ["ordinary_future","balanced_future","accuracy","balanced_accuracy"]};groups={g:protocol["feature_groups"][g] for g in protocol["feature_groups_used"]};rows=[];budgets=[];splits=[]
def label_set(mask,future=True):
 return {(int(s),int(t)+j) for s,t in zip(ss[mask],tt[mask]) for j in ([1,2,3,4] if future else [0])}
def matched_pool(train,reference):
 nblocks=len({(int(s),int(t)) for s,t in zip(ss[reference],tt[reference])});people=sorted(set(ss[train].tolist()));chosen=np.zeros(len(x),bool)
 for k,sid in enumerate(people):
  times=np.unique(tt[train&(ss==sid)]);quota=nblocks//len(people)+int(k<nblocks%len(people));assert quota<=len(times);pick=times[np.linspace(0,len(times)-1,quota,dtype=int)];chosen|=train&(ss==sid)&np.isin(tt,pick)
 assert chosen.sum()==reference.sum();assert len(label_set(chosen))==len(label_set(reference));return chosen
def fit(train,target,indices,alpha):
 counts={s:int((ss[train]==s).sum()) for s in set(ss[train])};sw=np.array([1/counts[s] for s in ss[train]]);sw/=sw.mean();scaler=StandardScaler().fit(x[train][:,indices]);model=Ridge(alpha=alpha).fit(scaler.transform(x[train][:,indices]),d[target][train],sample_weight=sw);return dict(indices=indices,scaler=scaler,model=model,alpha=alpha,target=target)
def predict(gate,signal):
 v=gate["model"].predict(gate["scaler"].transform(signal[:,gate["indices"]]));v[signal[:,3]==0,0]=0.;return v
folds=[]
for held in protocol["original_controllers"]:
 reference=original&(ss!=held);full=ss!=held
 for scheme,train in [("original3",reference),("expanded8",full),("expanded8_labelmatched",matched_pool(full,reference))]:folds.append((scheme,held,train,reference.sum()))
for held in protocol["controllers"]:
 ref=(original&(ss!=held)).sum() if held in protocol["original_controllers"] else original.sum()*.75;folds.append(("all9_LOPO",held,ss!=held,ref))
for scheme,held,train,nref in folds:
 test=ss==held;assert not(set(ss[train])&set(ss[test]));alpha=100*train.sum()/nref;signal=x[test];actions={"KEEP":np.zeros(test.sum(),int),"WRITE":np.full(test.sum(),2),"RESET":np.ones(test.sum(),int),"low_entropy_write":np.where(signal[:,0]<.5,2,0),"anchor_guard":np.where(signal[:,19]>.05,1,2)};labels=label_set(train);budgets.append(dict(scheme=scheme,held_subject=held,train_people=len(set(ss[train])),train_rows=int(train.sum()),unique_future_labels=len(labels),unique_current_labels=len(label_set(train,False)),ridge_alpha=float(alpha),young_train_people=len([s for s in set(ss[train]) if s<20]),middle_train_people=len([s for s in set(ss[train]) if s>=20])))
 for target in ["ordinary_future","ordinary_current","balanced_future","balanced_current"]:
  for group,indices in groups.items():
   gate=fit(train,target,indices,alpha);v=predict(gate,signal);name=target+"_"+group;actions[name]=np.column_stack([np.zeros(len(v)),v]).argmax(1);actions[name+"_write_only"]=np.where(v[:,1]>0,2,0);actions[name+"_reset_only"]=np.where(v[:,0]>0,1,0)
 for policy,action in actions.items():
  rows.append(dict(scheme=scheme,held_subject=held,policy=policy,**{k:float(v[test][np.arange(len(action)),action].mean()) for k,v in outcomes.items()},write_fraction=float((action==2).mean()),reset_fraction=float((action==1).mean())))
 splits.append(dict(scheme=scheme,held_subject=held,training_subjects=sorted(set(ss[train].tolist()))));print("DIVERSITY FOLD",scheme,held,int(train.sum()),alpha,flush=True)
people=pd.DataFrame(rows);people.to_csv(o/"loso_people.csv",index=False);summary=people.groupby(["scheme","policy"]).mean(numeric_only=True).drop(columns="held_subject").reset_index();summary.to_csv(o/"loso_summary.csv",index=False);pd.DataFrame(budgets).to_csv(o/"label_budgets.csv",index=False);(o/"loso_splits.json").write_text(json.dumps(splits,indent=2));fixed=["KEEP","WRITE","RESET","low_entropy_write","anchor_guard"];g=summary[summary.scheme=="all9_LOPO"];decisions=[]
for target in ["ordinary_future","balanced_future"]:
 best=float(g[g.policy.isin(fixed)][target].max())
 for _,r in g[g.policy.str.startswith(target)].iterrows():
  q=people[(people.scheme=="all9_LOPO")&(people.policy==r.policy)];positive=int((q[target]>0).sum());matched=r.policy.replace("_future_","_current_");other=g[g.policy==matched].iloc[0];decisions.append(dict(policy=r.policy,utility_target=target,utility=float(r[target]),best_fixed=best,positive_people=positive,matched_current_utility=float(other[target]),ordinary_ce=float(r.ordinary_future),balanced_ce=float(r.balanced_future),accuracy_gain_pp=100*float(r.accuracy),balanced_accuracy_gain=100*float(r.balanced_accuracy),advance=bool(r[target]>best and positive>=7)))
(o/"advance_decisions.json").write_text(json.dumps(dict(criterion=protocol["advance"],decisions=decisions),indent=2));(o/"gates").mkdir(exist_ok=True)
for target in ["ordinary_future","ordinary_current","balanced_future","balanced_current"]:
 for group,indices in groups.items():
  gate=fit(np.ones(len(x),bool),target,indices,100*len(x)/(original.sum()*.75));gate.update(training_subjects=protocol["controllers"],source_class_weights=weights.tolist(),features=group)
  with (o/"gates"/f"{target}_{group}.pkl").open("wb") as f:pickle.dump(gate,f)
(o/"audit.json").write_text(json.dumps(dict(status="PASS",source_cases=18,participants=9,pulses=len(x),label_disjoint_target_and_classifier_training_for_new_people=True,labelmatched_rows_and_unique_future_labels_exact=True,subject_disjoint_folds=True,regularization_scaled_by_training_size=True,future_probability_gains_recomputed=True,max_ce_error=maxerr,raw_future_labels_verified=True),indent=2))
adv=[r for r in decisions if r["advance"]];selected=summary[summary.policy.isin(fixed+["ordinary_future_with_embedding436","ordinary_current_with_embedding436","balanced_future_with_embedding436","balanced_current_with_embedding436","ordinary_future_direction_history36","balanced_future_direction_history36"])];style='<style>body{font:15px/1.65 system-ui;max-width:1450px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}section{overflow:auto}</style>'
html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>源控制器受试者多样性与标签数量对照</h1><p>固定混合来源分类器、拟合辅助头和LR0.01，复用阶段41四人轨迹，新增3/4/12/13/14五人。新增人员与本混合分类器训练、源验证和目标30人互斥，但已出现在其他研究分支，不能称新盲测。</p><p>对相同原四人的留一测试，分别用原3人、扩展8人、扩展8人且精确匹配原未来标签数量训练；按完整时间点块抽样、同时保留两条轨迹，不按收益或类别筛选。另作完整9人留一。Ridge正则随训练行数缩放，保持平均损失的正则强度；当前/未来同容量、类别平衡/普通CE并列，所有指标逐人等权。</p><p>新增五人均年轻，扩展训练池也改变了年龄和阶段组成，因此本实验检验更大参与者池的效果，不能单独识别纯粹的“人数效应”。当前标签与未来标签预算不同；若真实策略推进，仍需同未来标签预算的静态校准。</p>'+f'<p>18条轨迹、{len(x)}次脉冲，精确标签预算匹配和概率审计通过。完整9人留一满足预设推进条件的候选数：{len(adv)}。</p><h2>相同测试人员下的对照</h2><section>'+selected.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'</section><h2>完整九人推进判定</h2><section>'+pd.DataFrame(decisions).to_html(index=False,float_format=lambda v:f"{v:.7f}")+'</section><p><a href="label_budgets.csv">精确标签数量</a> · <a href="loso_people.csv">逐人结果</a> · <a href="audit.json">独立审计</a> · <a href="protocol.json">协议</a></p></html>';(o/"report.html").write_text(html);print("ADVANCE",len(adv),adv);print("COMPLETE",len(x))

from common import *
import pandas as pd,hashlib
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
O=ROOT/"phase117";O.mkdir(exist_ok=True)
names=["student_entropy","teacher_entropy","student_confidence","teacher_confidence","teacher_student_kl","soft_teacher_ce","probability_l1","topclass_agreement","student_margin","teacher_margin","proposed_update_norm"]+[f"student_p{i}" for i in range(5)]+[f"teacher_p{i}" for i in range(5)]
def observable(b,t,norm):
 assert b.shape==t.shape==(5,) and (b>0).all() and (t>0).all()
 bs=np.sort(b);ts=np.sort(t)
 return np.array([-sum(b*np.log(b)),-sum(t*np.log(t)),max(b),max(t),sum(t*(np.log(t)-np.log(b))),-sum(t*np.log(b)),sum(abs(t-b)),float(b.argmax()==t.argmax()),bs[-1]-bs[-2],ts[-1]-ts[-2],norm]+b.tolist()+t.tolist(),dtype=np.float64)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/"phase115";a=json.loads((source/"audit.json").read_text());assert a["status"]=="PASS"
pr=dict(purpose="Fixed source-only diagnostic of observable prediction of isolated teacherWRITE futureutility; no action policy, no online/target rollout.",training_people=[7,17,27,37],evaluation_people=[5,15,25,35],seeds=[17,29,43],pulses="Same62durationeligiblephase115pulses/person/seed; all3seeds equallyweighted, not independent people.",features=names,feature_information="Currentfrozenstudent/currentcausalteacher probabilities and proposedLNupdate norm only. No current/future true labels, future probability, timestamp, subject or seed ID in features.",primary_target="Mean balancedCEgain overnext4scoredwindows, sameweights phase36.",secondary_target="Mean ordinaryCEgain overnext4windows.",models="Sourcecontroller training mean; fixed ridgealpha10 on standardizedupdate-norm alone; fixed ridgealpha10 on21currentfeatures. Standardization training-only, scale floor1e-6, no feature selection/hyperparametersearch.",evaluation="Perperson/seed MSE andMSE improvement against fixed trainingmean; meanseed metricswithinperson then equal4people. Rank/linear correlations descriptive; no significance claim or gate.",label_budget="Future labels already read inphase115; 744 training pulse-seed rows, overlapping futurewindow occurrences, not newunique independent labels. Evaluation people previously observed in prior research, not blind.",implementation="Closed-form ridge with intercept after training centering, sklearn independentreplay.",input_hashes={f"seed{seed}_s{sid:02d}":sha(source/f"seed{seed}"/f"s{sid:02d}.npz") for seed in [17,29,43] for sid in [7,17,27,37,5,15,25,35]},cpu_ids=list(range(8,24)),gpu=False)
p=O/"protocol.json"
if p.exists():assert json.loads(p.read_text())==pr
else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
weights=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"])
rows=[];features=[]
for seed in pr["seeds"]:
 for group,sids in [("training",pr["training_people"]),("evaluation",pr["evaluation_people"])]:
  for sid in sids:
   d=np.load(source/f"seed{seed}"/f"s{sid:02d}.npz")
   for i,t in enumerate(d["times"]):
    yy=d["labels"][i,1:5];assert (yy>=0).all();bp=d["base"][i,1:5];wp=d["write"][i,1:5];delta=np.log(wp[np.arange(4),yy])-np.log(bp[np.arange(4),yy])
    features.append(observable(d["base"][i,0],d["teacher"][i,0],d["norms"][i]))
    rows.append(dict(seed=seed,group=group,subject=sid,t=int(t),balanced_ce_gain=float((delta*weights[yy]).mean()),ce_gain=float(delta.mean())))
df=pd.DataFrame(rows);X=np.array(features);train=(df.group=="training").to_numpy();assert train.sum()==744 and (~train).sum()==744 and np.isfinite(X).all()
mu=X[train].mean(0);scale=np.maximum(X[train].std(0),1e-6);Z=(X-mu)/scale;assert np.max(abs(Z[train].mean(0)))<1e-10
predictions=[];states={};replayerr=0.
for target in ["balanced_ce_gain","ce_gain"]:
 y=df[target].to_numpy();ym=float(y[train].mean());states[target+"_training_mean"]=ym
 for name,cols in [("training_mean",[]),("update_norm_ridge",[10]),("current_features_ridge",list(range(len(names))))]:
  if not cols:pred=np.full(len(y),ym)
  else:
   zz=Z[:,cols];tr=zz[train];beta=np.linalg.solve(tr.T@tr+10*np.eye(len(cols)),tr.T@(y[train]-ym));pred=ym+zz@beta;sk=Ridge(alpha=10).fit(tr,y[train]);err=float(abs(sk.predict(zz)-pred).max());assert err<1e-10;replayerr=max(replayerr,err);states[target+"_"+name]=dict(columns=cols,beta=beta.tolist(),intercept=ym)
  for i,r in df.iterrows():predictions.append(dict(**r.to_dict(),target=target,model=name,actual=float(y[i]),prediction=float(pred[i]),squared_error=float((y[i]-pred[i])**2),actual_positive=bool(y[i]>0),predicted_positive=bool(pred[i]>0)))
pd.DataFrame(predictions).to_csv(O/"predictions.csv.gz",index=False,compression="gzip")
states.update(feature_names=names,mean=mu.tolist(),scale=scale.tolist());(O/"models.json").write_text(json.dumps(states,indent=2))
results=[]
pred=pd.DataFrame(predictions)
for keys,d in pred.groupby(["group","subject","seed","target","model"]):
 group,sid,seed,target,model=keys;actual=d.actual.to_numpy();pp=d.prediction.to_numpy();base=float(states[target+"_training_mean"]);mse=float(np.mean((pp-actual)**2));bmse=float(np.mean((base-actual)**2));v=float(np.var(actual));corr=float(np.corrcoef(actual,pp)[0,1]) if np.std(pp)>1e-12 and np.std(actual)>1e-12 else np.nan;rank=float(spearmanr(actual,pp).statistic) if np.std(pp)>1e-12 else np.nan
 results.append(dict(group=group,subject=sid,seed=seed,target=target,model=model,mse=mse,constant_mse=bmse,mse_gain=bmse-mse,relative_mse_gain=1-mse/bmse if bmse else np.nan,r2=1-mse/v if v else np.nan,pearson=corr,spearman=rank,mean_actual=float(actual.mean()),mean_prediction=float(pp.mean()),actual_positive_rate=float((actual>0).mean()),predicted_positive_rate=float((pp>0).mean())))
ps=pd.DataFrame(results);ps.to_csv(O/"people_seeds.csv",index=False);metricscols=[c for c in ps if c not in ["group","subject","seed","target","model"]]
person=ps.groupby(["group","subject","target","model"])[metricscols].mean().reset_index();person.to_csv(O/"people.csv",index=False)
summary=person.groupby(["group","target","model"])[metricscols].mean().reset_index();positive=person.groupby(["group","target","model"]).mse_gain.apply(lambda x:int((x>0).sum())).reset_index(name="positive_mse_people");summary=summary.merge(positive,on=["group","target","model"]);summary.to_csv(O/"summary.csv",index=False)
audit=dict(status="PASS",feature_count=len(names),train_rows=int(train.sum()),evaluation_rows=int((~train).sum()),independent_ridge_prediction_error=replayerr,feature_inputs_current_only=True,source_training_standardization=True,no_threshold_or_policy_selection=True,no_online_or_target_rollout=True)
(O/"audit.json").write_text(json.dumps(audit,indent=2))
display=summary[summary.group=="evaluation"][["target","model","mse","mse_gain","relative_mse_gain","r2","pearson","spearman","positive_mse_people","mean_actual","mean_prediction"]]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>当前可观察信号能否预测单次WRITE的未来收益？</h1><p>固定源控制4人训练，源验证4人评价；这些人均已在此前研究中观察，所以是诊断而非盲验证。利用phase115已有1488次单独更新，训练744个种子-脉冲行、评价744行。三个种子的指标先在人内平均；未来窗重叠，不能当作独立个体。</p><p>仅使用当前学生/因果教师概率的21个固定特征及已提议更新范数；没有当前或未来真实标签、未来预测、时点编号、受试者或种子身份。主目标为未来4窗平衡CE收益，普通CE为次要目标。预测器固定为训练均值、仅更新范数岭回归、全部当前特征岭回归，alpha=10，无调参或特征选择。标准化只在训练4人拟合。</p>'+display.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p>MSE改善与固定源训练均值比较，正值更好；R²使用评价者自己的目标方差，可以为负。相关系数只作描述，不能替代误差及校准检查。特征模型若优于常数也不意味着策略优于直接教师/静态混合；本实验不选择阈值或动作，不启动在线或目标集更新。</p><p><a href="protocol.json">固定协议</a> · <a href="audit.json">独立解析/库实现审计</a> · <a href="people.csv">逐人结果</a></p>')
print(display.to_string(index=False));print(audit)

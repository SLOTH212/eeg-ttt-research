from common import *
import pandas as pd
OUT58=ROOT/"phase58";CW=np.array(json.loads((ROOT/"phase36/mixed_class_weights.json").read_text())["balanced_weights"])
def f1(cm):
 denominator=cm.sum(0)+cm.sum(1);return float(np.mean(np.divide(2*np.diag(cm),denominator,out=np.zeros(5),where=denominator>0)))
def cases():
 for phase,name in [(41,"vq_head"),(51,"spectral_scope"),(54,"current_meta")]:
  for sid in [7,17,27,37]:
   for trajectory in ["continuous","reset64"]:
    z=np.load(ROOT/f"phase{phase}"/f"lr0.01_{trajectory}_s{sid:02d}/pulses.npz");yield name,sid,trajectory,z["t"],z["future_labels"],np.stack([z["future_keep"],z["future_reset"],z["future_commit"]],axis=1).astype(np.float64)
 for variant in ["fresh_ordinary","fresh_balanced","states_ordinary","states_balanced"]:
  for sid in [7,17,27,37]:
   rows=json.loads((ROOT/"phase57/evaluation"/variant/f"s{sid:02d}.json").read_text())["rows"];fresh={r["t"]:r["base_probabilities"] for r in rows if r["state"]=="fresh"}
   for trajectory in ["continuous","reset64"]:
    selected=[r for r in rows if r["state"]==trajectory];times=np.array([r["t"] for r in selected]);y=np.array([r["future_labels"] for r in selected]);p=np.array([[r["base_probabilities"],fresh[r["t"]],r["write_probabilities"]] for r in selected]);yield "factorial_"+variant,sid,trajectory,times,y,p
def greedy(cm,allowed):
 n=len(cm);scores=[f1(cm[:,a].sum(0)) for a in allowed];initial=allowed[int(np.argmax(scores))];actions=np.full(n,initial,int);total=cm[:,initial].sum(0);start=f1(total);changes=0;passes=0
 for step in range(5):
  changed=0
  for i in range(n):
   current=actions[i];best=current;best_score=f1(total);best_total=total
   for a in allowed:
    if a==current:continue
    candidate=total-cm[i,current]+cm[i,a];score=f1(candidate)
    if score>best_score+1e-12:best=a;best_score=score;best_total=candidate
   if best!=current:actions[i]=best;total=best_total;changed+=1
  changes+=changed;passes=step+1
  if not changed:break
 assert f1(total)>=start-1e-12
 return actions,dict(starting_fixed_action=int(initial),passes=passes,action_changes=changes)
def run():
 rows=[];saved_actions={};checks=[];reset_by_point={};maxreset=0.
 for name,sid,trajectory,times,y,p in cases():
  n=len(times);assert p.shape==(n,3,4,5) and y.shape==(n,4);assert np.isfinite(p).all() and abs(p.sum(3)-1).max()<1e-5;root=ROOT/("data/processed" if sid<20 else "phase2/processed");truth=np.load(root/f"s{sid:02d}_meta.npz")["y"];assert np.array_equal(y,np.array([truth[t+1:t+5] for t in times]));pred=p.argmax(3);cm=np.zeros((n,3,5,5),dtype=int)
  for i in range(n):
   for a in range(3):cm[i,a]=np.bincount(y[i]*5+pred[i,a],minlength=25).reshape(5,5)
   key=(sid,int(times[i]))
   if key in reset_by_point:
    error=float(abs(p[i,1]-reset_by_point[key]).max());maxreset=max(maxreset,error);assert error<2e-5 and np.array_equal(p[i,1].argmax(1),reset_by_point[key].argmax(1))
   else:reset_by_point[key]=p[i,1]
  index=np.broadcast_to(y[:,None,:,None],(n,3,4,1));loss=-np.log(np.clip(np.take_along_axis(p,index,axis=3)[...,0],1e-12,1));ce=loss.mean(2);balanced=(loss*CW[y][:,None]).mean(2);correct=(pred==y[:,None]);count=correct.sum(2);weighted_correct=(correct*CW[y][:,None]).sum(2);basecm=cm[:,0].sum(0);basef1=f1(basecm);checks.append(dict(model=name,subject=sid,trajectory=trajectory,pulses=n))
  for restriction,allowed in [("write_only",[0,2]),("three",[0,1,2])]:
   policies={["KEEP","RESET","WRITE"][a]:np.full(n,a,int) for a in allowed};policies["CE_oracle"]=np.array(allowed)[ce[:,allowed].argmin(1)];policies["balanced_CE_oracle"]=np.array(allowed)[balanced[:,allowed].argmin(1)];policies["correct_oracle"]=np.array(allowed)[count[:,allowed].argmax(1)];policies["balanced_correct_oracle"]=np.array(allowed)[weighted_correct[:,allowed].argmax(1)];policies["MacroF1_greedy"],ginfo=greedy(cm,allowed);bestfixed=max(f1(cm[:,a].sum(0)) for a in allowed)
   for policy,actions in policies.items():
    ii=np.arange(n);total=cm[ii,actions].sum(0);score=f1(total);pr=pred[ii,actions];chosen_correct=pr==y;b=correct[:,0];ce_gain=float((ce[:,0]-ce[ii,actions]).mean());balanced_gain=float((balanced[:,0]-balanced[ii,actions]).mean());acc=float((chosen_correct.astype(float)-b).mean()*100)
    if policy=="CE_oracle":assert ce_gain>=-1e-12
    if policy=="correct_oracle":assert acc>=0
    if policy=="MacroF1_greedy":assert score>=bestfixed-1e-12
    rows.append(dict(model=name,subject=sid,trajectory=trajectory,restriction=restriction,policy=policy,pulses=n,sample_macro_f1=score*100,macro_f1_gain_vs_keep_pp=(score-basef1)*100,macro_f1_gain_vs_best_fixed_pp=(score-bestfixed)*100,accuracy_gain_pp=acc,ce_gain=ce_gain,balanced_ce_gain=balanced_gain,corrected=int((~b&chosen_correct).sum()),harmed=int((b&~chosen_correct).sum()),changed=int((pr!=pred[:,0]).sum()),write_fraction=float((actions==2).mean()),reset_fraction=float((actions==1).mean()))) 
    saved_actions[f"{name}_s{sid:02d}_{trajectory}_{restriction}_{policy}"]=actions.astype(np.int8)
 d=pd.DataFrame(rows);d.to_csv(OUT58/"case_results.csv",index=False);columns=["sample_macro_f1","macro_f1_gain_vs_keep_pp","macro_f1_gain_vs_best_fixed_pp","accuracy_gain_pp","ce_gain","balanced_ce_gain","write_fraction","reset_fraction"];per=d.groupby(["model","subject","restriction","policy"])[columns].mean().reset_index();per.to_csv(OUT58/"people.csv",index=False);summary=per.groupby(["model","restriction","policy"])[columns].mean();summary.to_csv(OUT58/"summary.csv");np.savez_compressed(OUT58/"actions.npz",**saved_actions);key=summary.reset_index();key=key[key.policy.isin(["WRITE","CE_oracle","correct_oracle","MacroF1_greedy"])];key.to_csv(OUT58/"key_comparisons.csv",index=False)
 (OUT58/"audit.json").write_text(json.dumps(dict(status="PASS",cases=len(checks),unique_people=4,models=7,action_blocks=sum(v["pulses"] for v in checks),stored_decision_vectors=len(saved_actions),fixed_five_class_f1_from_confusions=True,same_task_labels_checked=True,source_reset_cross_model_max_probability_error=maxreset,oracle_CE_never_worse_inCE=True,oracle_correct_never_worse_inCorrectCount=True,greedy_f1_never_worse_than_best_fixed_on_sample=True,strict_global_upper_bound_claim=False,cases_detail=checks),indent=2))
 html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1400px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>后见损失收益与任务指标空间</h1><p>对七种既有写入机制、四名来源控制器受试者、两条固定参数轨迹，使用已保存的KEEP/RESET/WRITE未来四窗预测，分别按交叉熵、平衡交叉熵、正确数、平衡正确数作后见选择；另外从最佳固定动作出发，最多五轮坐标改进样本MacroF1。每次脉冲必须给四窗共同选择同一个动作，不逐窗口拆分动作。</p><p>所有选择都读取真实未来标签，仅为离线诊断，不是可部署方法。MacroF1贪心选择不是全局最优，不是严格上界；各脉冲继承固定参考轨迹且分叉后暂停更新，所以样本分数也不是这些动作在完整闭环中的可实现成绩。最佳固定动作是每名受试者/每条轨迹按真实标签选择的后见最佳固定动作，比一个统一固定策略更强；同时报告相对KEEP的变化。两轨迹先在人内平均，再按四名受试者等权，不能把脉冲当独立人群。</p><p>WRITE-only空间用于检查新增参数写入是否能改善真实任务指标；三动作空间还包含修复历史损害的RESET。CE后见选择只保证CE不差，正确数选择只保证正确数不差，均不保证MacroF1。不同辅助机制的源监督成本不同，本表不作成本公平的部署排名。</p>'+key.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p><a href="case_results.csv">全部逐人逐轨迹结果</a> · <a href="summary.csv">完整汇总</a> · <a href="actions.npz">固定后见动作</a> · <a href="audit.json">独立一致性审计</a> · <a href="protocol.json">协议</a></p></html>';(OUT58/"report.html").write_text(html);print(key[["model","restriction","policy","macro_f1_gain_vs_best_fixed_pp","accuracy_gain_pp","ce_gain"]].to_string(index=False))
if __name__=="__main__":run()

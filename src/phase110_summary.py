from common import *
from phase2_temporal import ema
import pandas as pd
O=ROOT/"phase110";pr=json.loads((O/"protocol.json").read_text());assert json.loads((O/"audit.json").read_text())["status"]=="PASS";rows=[]
for sid in pr["subjects"]:
 folder=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(folder/f"s{sid:02d}_meta.npz")["y"];z=np.load(O/"predictions"/f"s{sid:02d}.npz")
 for key in z.files:
  method,seed=key.rsplit("_cbramod_seed",1)
  for alpha in pr["ema"]:
   p=z[key] if alpha==1 else ema(z[key],alpha);mm=metrics(y,p);cm=np.array(mm["confusion"]);rows.append(dict(subject=sid,method=method,seed=int(seed),ema=alpha,macro_f1=100*mm["macro_f1"],accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"]))
new=pd.DataFrame(rows);assert len(new)==144;df=pd.concat([pd.read_csv(ROOT/"phase109/participants.csv"),new],ignore_index=True);assert len(df)==360;df.to_csv(O/"participants.csv",index=False);person=df.groupby(["subject","method","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();person.to_csv(O/"seedmean_people.csv",index=False);summary=person.groupby(["method","ema"])[["macro_f1","accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
pairs=[]
for short in ["guard","floor"]:
 pairs.extend([(f"corrupted_reset_{short}",f"corrupted_{short}","reset_on_corrupted"),(f"reference_reset_{short}",f"reference_{short}","sham_reset_on_reference"),(f"corrupted_reset_{short}",f"reference_reset_{short}","remaining_shift_penalty_with_reset")])
comp=[]
for alpha,g in person.groupby("ema"):
 p=g.pivot(index="subject",columns="method",values="macro_f1")
 for new,old,label in pairs:
  d=p[new]-p[old];comp.append(dict(ema=alpha,comparison=label,new_method=new,old_method=old,people=len(d),new_f1=p[new].mean(),old_f1=p[old].mean(),difference_pp=d.mean(),positive_people=int((d>0).sum()),worst_pp=d.min()))
comp=pd.DataFrame(comp);comp.to_csv(O/"comparisons.csv",index=False)
audit=dict(status="PASS",new_streams=144,total_with_reference=360,within_person_seedmetric_averaging=True,sham_reset_control=True,synthetic_boundaries_known=True,no_deployable_reset_detector=True,no_gradientTTT=True)
(O/"summary_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>已知人工变点的输入状态重置诊断</h1><p>沿用phase109中间三分之一幅值乘0.1的固定实验，仅在两个人工边界重置原有logRMS估计器。未扰动记录也在相同边界做sham reset，以区分重置本身与幅值变化后的历史状态影响。原guard/floor公式、阈值、EMA和模型均不变。</p>'+comp[comp.ema==.5].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>全部EMA与对照</h2>'+comp.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>边界来自人工实验安排，实际部署并不知道真实幅值变化时刻。本结果既不是可部署变点检测，也不是全局性能上界，更不是梯度TTT或原提案的参数保留策略成功。源记录预热段全为清醒，恢复段主要为清醒；低幅值中段覆盖睡眠各阶段，不能把恢复结果外推到全部阶段。</p><p><a href="audit.json">输入前缀与冻结模型审计</a> · <a href="summary_audit.json">三种子统计审计</a> · <a href="../phase109/report.html">保留未重置对照与响应曲线</a></p>')
print(comp[comp.ema==.5].to_string(index=False));print(audit)

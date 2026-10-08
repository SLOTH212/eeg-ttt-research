
import os
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"phase132"; OUT.mkdir(exist_ok=True)
files=[ROOT/"phase124/people_transitions.csv",ROOT/"phase124/people_temporal.csv"]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={"purpose":"Descriptive exact decomposition of equal-person mean adjacent transition rate","inputs":{str(p):sha(p) for p in files},"reference":"training","formula":"delta R = sum((pi_t-pi_s)*(q_t+q_s)/2) + sum((q_t-q_s)*(pi_t+pi_s)/2)","weighting":"Normalize each person's joint transition counts to sum one, then average people within cohort","scope":"Retrospective descriptive labels only; no causal identification, model selection, retraining, crop, or gate"}
p=OUT/"protocol.json"
if p.exists(): assert json.loads(p.read_text())==protocol
else:p.write_text(json.dumps(protocol,indent=2))
d=pd.read_csv(files[0]); people=pd.read_csv(files[1])
stages=["W","N1","N2","N3","REM"]
joints={}; errors=[]; npeople={}; stage_rows=[]
for group,g in d.groupby("group"):
 mats=[]
 for sid,s in g.groupby("subject"):
  assert len(s)==25
  counts=s.pivot(index="from_stage",columns="to_stage",values="count").loc[stages,stages].to_numpy(float)
  assert counts.sum()>0
  m=counts/counts.sum(); mats.append(m)
  r=m.sum()-np.trace(m)
  gold=people.loc[(people.group==group)&(people.subject==sid),"adjacent_change_fraction"].item()
  errors.append(abs(r-gold))
 J=np.mean(mats,axis=0); joints[group]=J; npeople[group]=len(mats)
 pi=J.sum(1); assert (pi>0).all()
 q=(pi-np.diag(J))/pi
 for i,stage in enumerate(stages):
  stage_rows.append(dict(group=group,stage=stage,people=len(mats),origin_stage_mass=pi[i],conditional_change=q[i],transition_contribution=pi[i]*q[i]))
base=joints["training"]; ps=base.sum(1); qs=(ps-np.diag(base))/ps
rows=[]; detail=[]
for group,J in joints.items():
 if group=="training":continue
 pt=J.sum(1); qt=(pt-np.diag(J))/pt
 comp=(pt-ps)*(qt+qs)/2; within=(qt-qs)*(pt+ps)/2
 rs=float(ps@qs); rt=float(pt@qt); delta=rt-rs
 err=abs(delta-comp.sum()-within.sum()); errors.append(err)
 rows.append(dict(group=group,people=npeople[group],source_rate=rs,target_rate=rt,gap=delta,stage_composition_component=float(comp.sum()),within_stage_component=float(within.sum()),source_mix_target_rates=float(ps@qt),target_mix_source_rates=float(pt@qs),identity_error=err))
 for i,stage in enumerate(stages):
  detail.append(dict(group=group,stage=stage,composition_component=comp[i],within_stage_component=within[i],total_component=comp[i]+within[i]))
assert max(errors)<1e-12
df=pd.DataFrame(rows);df.to_csv(OUT/"decomposition.csv",index=False)
pd.DataFrame(stage_rows).to_csv(OUT/"stage_rates.csv",index=False)
pd.DataFrame(detail).to_csv(OUT/"stage_components.csv",index=False)
audit={"status":"PASS","people":sum(npeople.values()),"groups":npeople,"max_person_rate_or_decomposition_error":max(errors),"no_new_predictions_or_training":True,"input_hashes_rechecked":all(sha(p)==protocol["inputs"][str(p)] for p in files)}
assert audit["input_hashes_rechecked"]
(OUT/"audit.json").write_text(json.dumps(audit,indent=2))
display=df.copy()
for c in display.columns:
 if c not in ["group","people","identity_error"]:display[c]*=100
html='<!doctype html><meta charset="utf-8"><title>Phase132 时间结构分解</title><style>body{font-family:sans-serif;max-width:1250px;margin:40px auto;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}</style><h1>阶段转换率差异的描述性分解</h1><p>每名受试者的 5×5 相邻转换计数先归一化，再在人群内等权平均。由该联合矩阵推导起始阶段占比 π 和各阶段转出概率 q，使总转换率 R=Σπq 精确成立。这里 π 是有效相邻窗口起点的阶段分布，与单窗口阶段占比略有不同。</p><p>差值被对称分为阶段占比项 Σ(π目标−π源)(q目标+q源)/2，以及阶段内转换项 Σ(q目标−q源)(π目标+π源)/2。分解是统计恒等式，不证明因果，也不区分生理差异、记录方式或评分差异。</p><p>下表率和分解项使用百分比/百分点；每人等权。未知阶段相邻对延续 Phase124 的排除规则，没有重新裁剪标签、拟合模型或设计在线门控。标准化两列只是算术替换分布，非临床反事实。</p>'+display.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p>逐人转换率回算与分解最大误差：'+str(max(errors))+'</p>'
(OUT/"report.html").write_text(html)
print(df.to_string(index=False));print(json.dumps(audit))

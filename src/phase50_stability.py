from common import *
import pandas as pd
o=ROOT/"phase50";d=pd.read_csv(ROOT/"phase48/group_people.csv");g=d[(d["mode"]=="masked")&(d.objective=="spectral")];people=sorted(g.subject.unique());rng=np.random.default_rng(5001);indices=rng.integers(0,len(people),size=(10000,len(people)));rows=[]
for target in ["clipped_future_dot","clipped_current_dot"]:
 a=g.pivot(index="subject",columns="group",values=target).reindex(people).to_numpy();boot=a[indices].mean(1)
 for group in range(13):
  v=a[:,group];lo,hi=np.percentile(boot[:,group],[2.5,97.5]);rows.append(dict(target=target,group=group,mean=float(v.mean()),positive_people=int((v>0).sum()),bootstrap_positive_fraction=float((boot[:,group]>0).mean()),bootstrap_low=float(lo),bootstrap_high=float(hi)))
df=pd.DataFrame(rows);df.to_csv(o/"scope_stability.csv",index=False);(o/"scope_stability.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:30px auto}td,th{padding:5px}</style><h1>源侧选层稳定性描述</h1><p>在阶段50选层和在线协议锁定后，对九名控制器训练受试者按人bootstrap10000次，描述逐组一阶内积的变异。此表不用于重新选层、调整阈值或步长；不包含源验证或目标受试者。区间只反映既有九人的抽样变异，不包含分类器训练种子和多轮探索的不确定性。一阶内积不等于实际在线收益。</p>'+df.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'</html>');print(df.to_string(index=False))

from common import *
import pandas as pd
o=ROOT/"phase52";rows=[]
for variant in ["initial","future","current","reconstruction","initial_normmatched"]:
 for sid in [7,17,27,37]:
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");y=np.load(root/f"s{sid:02d}_meta.npz")["y"]
  for r in json.loads((o/"evaluation"/variant/f"s{sid:02d}.json").read_text())["rows"]:
   t=r["t"];yy=np.array(r["future_labels"]);bp=np.array(r["future_base_probabilities"]);wp=np.array(r["future_write_probabilities"]);b=bp.argmax(1)==yy;w=wp.argmax(1)==yy;rows.append(dict(variant=variant,subject=sid,t=t,current_stage=int(y[t]),transition="change_within4" if np.any(yy!=y[t]) else "stable4",future_ce_gain=r["future_ce_gain"],balanced_future_ce_gain=r["balanced_future_ce_gain"],accuracy_gain_pp=float((w.astype(float)-b).mean()*100),corrected=int((~b&w).sum()),harmed=int((b&~w).sum()),changed=r["future_argmax_changes"]))
d=pd.DataFrame(rows);d.to_csv(o/"transition_pulses.csv",index=False);tables=[]
for grouping in ["transition","current_stage"]:
 per=d.groupby(["variant","subject",grouping]).agg(pulses=("t","size"),future_ce_gain=("future_ce_gain","mean"),balanced_future_ce_gain=("balanced_future_ce_gain","mean"),accuracy_gain_pp=("accuracy_gain_pp","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),changed=("changed","sum")).reset_index();per.to_csv(o/(grouping+"_people.csv"),index=False)
 summary=per.groupby(["variant",grouping]).agg(people=("subject","nunique"),pulses=("pulses","sum"),future_ce_gain=("future_ce_gain","mean"),balanced_future_ce_gain=("balanced_future_ce_gain","mean"),accuracy_gain_pp=("accuracy_gain_pp","mean"),corrected=("corrected","sum"),harmed=("harmed","sum"),changed=("changed","sum"));summary.to_csv(o/(grouping+"_summary.csv"));tables.append(summary)
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1200px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>元辅助头的阶段转变诊断</h1><p>阶段52结果完成后的离线机制分析，使用已评分256个持出时点，未重新训练、选阈值或改变在线动作。按未来四窗是否出现不同于当前真实阶段的标签分组，再按当前阶段分组（0W/1N1/2N2/3N3/4REM）。真实边界仅用于这里的离线评分，部署无法访问。</p><p>先在每名受试者的每个组内平均，再等人汇总；同时给出有该分组的人数和脉冲总数。组间样本和人员构成不同，不能把差异作因果解释，也不把相邻窗口当独立样本做显著性检验。当前元头三个改对一个改错的总量很小。</p>'+''.join(t.to_html(float_format=lambda v:f"{v:.7f}") for t in tables)+'</html>';(o/"transition_analysis.html").write_text(html);print(tables[0].to_string())

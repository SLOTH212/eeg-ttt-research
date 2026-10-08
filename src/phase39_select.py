from common import *
from phase2_temporal import ema
import pandas as pd,hashlib
o=ROOT/"phase39";rows=[]
assert json.loads((ROOT/"phase37/audit.json").read_text())["complete"]
for objective,variant,root in [("masked_token","fitted",ROOT/"phase37"),("masked_token","unfitted",ROOT/"phase37"),("entropy","unfitted",o)]:
 for lr in [.0001,.001,.01]:
  case=root/f"mixed_{variant}_{lr:g}";meta=json.loads((case/"complete.json").read_text());lock=json.loads((case/"selection.json").read_text());checkpoint=ROOT/"phase35/mixed/model.pt" if variant=="fitted" else ROOT/"phase28/mixed_joint_0/model.pt";assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==lock["checkpoint_sha256"]
  for sid in [5,15,25,35]:
   z=np.load(case/f"s{sid:02d}.npz");rec=json.loads((case/f"s{sid:02d}.json").read_text());base=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz");assert np.array_equal(z["y"],base["y"]);assert np.isfinite(z["p"]).all() and abs(z["p"].sum(1)-1).max()<1e-5;assert rec["nonwritable_unchanged"] and rec["radius_verified"];mm=metrics(z["y"],z["p"]);assert abs(mm["macro_f1"]-rec["metrics"]["macro_f1"])<1e-12;rows.append(dict(objective=objective,variant=variant,lr=lr,subject=sid,f1=mm["macro_f1"],frozen=rec["frozen"]["macro_f1"],weak_ema=rec["weak_ema"]["macro_f1"],strong_ema=rec["strong_ema"]["macro_f1"]))
d=pd.DataFrame(rows);d.to_csv(o/"participants.csv",index=False);summary=d.groupby(["objective","variant","lr"]).mean(numeric_only=True).drop(columns="subject").reset_index();summary.to_csv(o/"summary.csv",index=False);decisions=[]
for objective,variant in [("masked_token","fitted"),("masked_token","unfitted"),("entropy","unfitted")]:
 g=summary[(summary.objective==objective)&(summary.variant==variant)];best=g.sort_values(["f1","lr"],ascending=[False,True]).iloc[0];baseline=float(g.frozen.iloc[0]);lr=float(best.lr) if best.f1>baseline else 0.;decisions.append(dict(objective=objective,variant=variant,selected_lr=lr,selected_f1=max(float(best.f1),baseline),frozen_f1=baseline))
static=summary[["frozen","weak_ema","strong_ema"]].iloc[0];best_static=max(static.index,key=lambda k:static[k]);selection=dict(complete=True,source="mixed",decisions=decisions,best_static=best_static,best_static_f1=float(static[best_static]),validation_subjects=[5,15,25,35],grid=[.0001,.001,.01],freeze_included=True,primary_metric="person-mean Macro-F1",source_validation_only=True)
p=o/"source_selection.json"
if p.exists():assert json.loads(p.read_text())==selection
else:p.write_text(json.dumps(selection,indent=2))
(o/"audit.json").write_text(json.dumps(dict(status="PASS",complete=True,source_adapted_streams=36,new_entropy_streams=12,recomputed_metrics=True,checkpoint_hashes_verified=True),indent=2))
style='<style>body{font:15px/1.6 system-ui;max-width:1100px;margin:30px auto}td,th{padding:6px;border-bottom:1px solid #ddd}</style>'
(o/"report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>混合来源：迁移前补齐熵更新对照</h1><p>同一冻结分类器，同样3个步长0.0001/0.001/0.01和不更新候选，在四名既有源验证受试者上分别选择拟合辅助头掩码、未拟合辅助头掩码和熵更新。源分数锁定后才开始本分支目标评估；原始Macro-F1选择步长，平滑后结果作为固定次要比较。</p>'+summary.to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<h2>锁定选择</h2>'+pd.DataFrame(decisions).to_html(index=False,float_format=lambda x:f"{x:.6f}")+'<p><a href="source_selection.json">锁定源选择</a> · <a href="audit.json">审计</a></p></html>')
print(selection)

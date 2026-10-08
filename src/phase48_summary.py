from common import *
import pandas as pd,hashlib
o=ROOT/"phase48";layout=json.loads((o/"parameter_layout.json").read_text());columns={g:np.concatenate([np.arange(p["offset"],p["offset"]+p["length"]) for p in layout["parameters"] if p["group"]==g]) for g in range(13)};frames=[];doterr=0.;normerr=0.;rows=0
for mode in ["masked","unmasked"]:
 assert (o/mode/"complete.json").exists()
 for sid in [3,4,7,12,13,14,17,27,37]:
  obj=json.loads((o/mode/f"s{sid:02d}.json").read_text());assert obj["parameters_unchanged"];df=pd.DataFrame(obj["rows"]);grads=np.load(o/mode/f"s{sid:02d}_gradients.npz");ref=json.loads((ROOT/"phase47"/mode/f"s{sid:02d}.json").read_text());reference={(r["t"],r["objective"]):r for r in ref["rows"] if r["lr"]==.001}
  for objective,g in df.groupby("objective"):
   a=grads[objective];assert a.shape==(len(grads["times"]),10960) and np.isfinite(a).all()
   for group in range(13):
    expected=g[g.group==group].set_index("t").loc[grads["times"]].group_gradient_squared_norm.to_numpy();observed=np.square(a[:,columns[group]].astype(np.float64)).sum(1);error=np.max(abs(observed-expected)/np.maximum(abs(expected),1e-10));normerr=max(normerr,float(error));assert error<2e-5
   totals=g.groupby("t").clipped_future_dot.sum()
   for t,v in totals.items():
    expected=reference[int(t),objective]["first_order_gain"]/.001;error=abs(v-expected);doterr=max(doterr,error);assert error<max(1e-5,abs(expected)*1e-4)
  frames.append(df);rows+=len(df)
d=pd.concat(frames,ignore_index=True);per=d.groupby(["mode","objective","subject","group","group_name"])[["clipped_future_dot","clipped_current_dot","group_gradient_squared_norm"]].mean().reset_index();per.to_csv(o/"group_people.csv",index=False);summary=per.groupby(["mode","objective","group","group_name"]).mean(numeric_only=True).drop(columns="subject").reset_index();summary.to_csv(o/"group_summary.csv",index=False);scopes=[]
for (mode,objective),g in per.groupby(["mode","objective"]):
 for held in sorted(g.subject.unique()):
  train=g[g.subject!=held];assert held not in train.subject.tolist();future=train.groupby("group").clipped_future_dot.mean().reindex(range(13));current=train.groupby("group").clipped_current_dot.mean().reindex(range(13));scopes.append(dict(mode=mode,objective=objective,held_subject=int(held),training_subjects=sorted(set(train.subject.astype(int).tolist())),future_groups=np.flatnonzero(future.to_numpy()>0).tolist(),current_groups=np.flatnonzero(current.to_numpy()>0).tolist(),future_training_dots=future.tolist(),current_training_dots=current.tolist()))
target=ROOT/"phase49";target.mkdir(exist_ok=True);data=dict(source_only=True,selection_rule="Select groups with strictly positive equally weighted training-person mean globally-clipped first-order dot; exclude held person",groups=layout["group_names"],folds=scopes,source_gradient_layout_sha256=hashlib.sha256((o/"parameter_layout.json").read_bytes()).hexdigest());path=target/"scopes.json"
if path.exists():assert json.loads(path.read_text())==data
else:path.write_text(json.dumps(data,indent=2))
(o/"audit.json").write_text(json.dumps(dict(status="PASS",group_rows=rows,groups=13,writable_parameters=10960,maximum_group_gradient_norm_relative_error=normerr,maximum_total_dot_error=doterr,stored_current_gradients_verified=True,subject_disjoint_scope_training=True),indent=2));style='<style>body{font:15px/1.65 system-ui;max-width:1150px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style>';(o/"report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>按层分解新鲜更新方向</h1><p>相同573个源时点、不改变任何参数，将10960个LayerNorm参数分成12个Transformer块及最终归一化共13组。下表是当前辅助梯度与未来/当前监督任务梯度内积，沿用原全梯度clip系数，按人等权。正值只表示一阶近似有利，不是实际分类收益。</p><p>保存的当前辅助梯度范数与逐组记录独立复算一致，逐组未来内积之和与阶段47总一阶收益吻合。后续按人留一拟合正内积组掩码，比较预定固定层范围、同容量当前标签选择及同更新范数的全梯度方向对照；持出人的标签不参与其范围选择。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p><a href="audit.json">独立审计</a> · <a href="../phase49/scopes.json">持出人互斥的范围选择</a></p></html>');print(summary.to_string(index=False));print("AUDIT",rows,doterr,normerr)

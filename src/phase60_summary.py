from common import *
from phase2_temporal import ema
import pandas as pd, hashlib
o=ROOT/"phase60"; pr=json.loads((o/"protocol.json").read_text())
assert (o/"execution_complete.json").exists()
assert json.loads((o/"graph_audit.json").read_text())["status"]=="PASS"
rows=[]; checks=[]; modes=pr["existing_modes"]+pr["new_modes"]
for v in pr["variants"]:
 assert hashlib.sha256((ROOT/"phase52"/v/"model.pt").read_bytes()).hexdigest()==pr["model_sha256"][v]
for sid in pr["validation_subjects"]:
 b=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz"); y=b["y"]; source=b["p"]
 def add(name,v,mode,blend,smooth,p):
  mm=metrics(y,p)
  rows.append(dict(subject=sid,method=name,variant=v,mode=mode,adapted_fraction=blend,ema=smooth,macro_f1=100*mm["macro_f1"],balanced_accuracy=100*mm["balanced_accuracy"],cross_entropy=mm["ce"]))
 for smooth in [1.,.8,.5]:
  add(f"frozen_ema{smooth:g}","frozen","frozen",0,smooth,source if smooth==1 else ema(source,smooth))
 for v in pr["variants"]:
  z=np.load(ROOT/"phase53"/("static_"+v)/f"s{sid:02d}.npz"); assert np.array_equal(z["y"],y)
  for smooth in [1.,.8,.5]:
   p=z["predictions"]; add(f"static_{v}_ema{smooth:g}",v,"static",0,smooth,p if smooth==1 else ema(p,smooth))
  for mode in modes:
   folder=o/f"{v}_{mode}"; assert (folder/"complete.json").exists()
   z=np.load(folder/f"s{sid:02d}.npz"); meta=json.loads((folder/f"s{sid:02d}.json").read_text()); p=z["p"]
   assert np.array_equal(z["y"],y) and np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-5
   assert np.isfinite(z["loss"]).all() and np.isfinite(z["gradient_norm"]).all()
   assert meta["nonwritable_unchanged"] and meta["radius_verified"]
   assert abs(p[0]-source[0]).max()<2e-5
   assert abs(metrics(y,p)["macro_f1"]-meta["metrics"]["macro_f1"])<1e-12
   new=not folder.is_symlink()
   if new:
    mask=z["gradient_step"]; n=min(int(mode[6:]),len(y)) if mode.startswith("warmup") else len(y)
    assert mask.sum()==n==meta["gradient_steps"]
    assert mask[:n].all() and not mask[n:].any()
    assert z["final_params"].shape==(10960,) and np.isfinite(z["final_params"]).all()
    if mode.startswith("warmup"):
     assert meta["warmup_state_held_exact"]
     assert (z["loss"][n:]==0).all() and (z["gradient_norm"][n:]==0).all()
   checks.append(dict(subject=sid,variant=v,mode=mode,new=new))
   for blend in [.25,.5,.75,1.]:
    mixed=blend*p+(1-blend)*source
    for smooth in [1.,.8,.5]:
     add(f"{v}_{mode}_blend{blend:g}_ema{smooth:g}",v,mode,blend,smooth,mixed if smooth==1 else ema(mixed,smooth))
df=pd.DataFrame(rows); df.to_csv(o/"participants.csv",index=False)
summary=df.groupby(["method","variant","mode","adapted_fraction","ema"],as_index=False)[["macro_f1","balanced_accuracy","cross_entropy"]].mean().sort_values("macro_f1",ascending=False)
summary.to_csv(o/"summary.csv",index=False)
comparisons=[]
for v in pr["variants"]:
 for smooth in [1.,.8,.5]:
  adapt=summary[(summary.variant==v)&(~summary["mode"].isin(["static"]))&(summary.ema==smooth)]
  best=adapt.iloc[0]; static=summary[(summary.variant==v)&(summary["mode"]=="static")&(summary.ema==smooth)].iloc[0]
  dd=df[df.method==best.method].set_index("subject").macro_f1-df[df.method==static.method].set_index("subject").macro_f1
  comparisons.append(dict(variant=v,ema=smooth,best_adaptive=best.method,adaptive_f1=best.macro_f1,matched_static_f1=static.macro_f1,difference_pp=best.macro_f1-static.macro_f1,positive_people=int((dd>0).sum())))
cmp=pd.DataFrame(comparisons); cmp.to_csv(o/"matched_comparisons.csv",index=False)
raw=summary[(summary.adapted_fraction==1)&(summary.ema==1)]
audit=dict(status="PASS",trajectories=len(checks),new_trajectories=sum(c["new"] for c in checks),method_streams=len(df),graph_eager_cases=8,model_sha_verified=True,warmup_exact_hold_verified=True,alpha_zero_deduplicated_to_frozen=True,source_only=True,no_automatic_target_advance=True)
(o/"audit.json").write_text(json.dumps(audit,indent=2))
html='<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:15px/1.7 system-ui;max-width:1400px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style><h1>简单记忆时长、固定遗忘与来源预测混合</h1><p>源侧已选当前头步长1e-4、未来头步长0.01；同一冻结基础分类器。仅前64/256个窗口更新，此后参数严格保持；固定遗忘每步将参数差乘0.99/0.999。48条轨迹中36条新运行、12条复用。所有预测先于当前更新。混合权重固定，不按测试标签选择；来源模型混合在线需要额外冻结前向，缓存回放不代表免费计算。</p><p>四位源验证受试者已反复使用。下表为探索性源网格描述，没有新的盲测；各头分别与317/1268个完全匹配标签的静态校准比较。不同额外标签预算的头不能直接称公平优劣。所有方法使用相同输出EMA网格，alpha=0复用冻结。</p><h2>相同EMA下比较</h2>'+cmp.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<h2>原始在线输出</h2>'+raw.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<h2>全部源网格</h2>'+summary.to_html(index=False,float_format=lambda x:f"{x:.5f}")+'<p>仅描述简单基线覆盖的范围，本阶段不自动晋级目标人群。暖启动后的0损失记录表示未计算，由gradient_step掩码区分。计算时间排除共享预处理；梯度步数不等于非零有效更新数。</p><p><a href="protocol.json">协议</a> · <a href="audit.json">审计</a> · <a href="graph_audit.json">CUDA/eager一致性</a></p></html>'
(o/"report.html").write_text(html)
print(cmp.to_string(index=False));print(raw.to_string(index=False));print(audit)

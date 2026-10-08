from common import *
import pandas as pd
o=ROOT/"phase74";assert (o/"complete.json").exists();layout=json.loads((o/"layout.json").read_text());audit=json.loads((o/"execution_audit.json").read_text());assert audit["state_unchanged"]
fd=pd.DataFrame(audit["finite_differences"]);fd["relative_error"]=fd.absolute_error/np.maximum(abs(fd.analytical),1e-6);fd["passes"]=fd.absolute_error<=2e-4+.02*abs(fd.analytical);fd.to_csv(o/"finite_differences.csv",index=False);assert fd.passes.all()
rows=[];full=[];replay=[];coverage=[]
for sid in [7,17,27,37]:
 a=np.load(o/f"s{sid:02d}_gradients.npz");truth=np.load(ROOT/("data/processed" if sid<20 else "phase2/processed")/f"s{sid:02d}_meta.npz")["y"];pulses=json.loads((ROOT/"phase73"/f"s{sid:02d}.json").read_text())["rows"];lookup={(r["t"],r["lr"],r["variant"]):r for r in pulses}
 for i,t in enumerate(a["times"]):
  y=a["labels"][i];assert np.array_equal(y,truth[t:t+5]);transition=bool((y[1:]!=y[0]).any());coverage.append(dict(subject=sid,t=int(t),current_stage=int(y[0]),transition=transition))
  for objective in ["waveform","entropy"]:
   g=a[objective][i].astype(float);norm=np.linalg.norm(g);assert np.isfinite(g).all();clip=min(1,1/(norm+1e-6))
   for target in ["current_ce","future_ce","future_balanced_ce"]:
    h=a[target][i].astype(float);dot=float(g@h);cos=dot/max(norm*np.linalg.norm(h),1e-30)
    full.append(dict(subject=sid,t=int(t),current_stage=int(y[0]),transition=transition,objective=objective,target=target,dot=dot,cosine=cos,linear_gain_lr001=.01*clip*dot))
    for layer in range(12):
     idx=np.concatenate([np.arange(v["offset"],v["offset"]+v["length"]) for v in layout if v["layer"]==layer]);gl=g[idx];hl=h[idx]
     rows.append(dict(subject=sid,t=int(t),objective=objective,target=target,layer=layer,dot=float(gl@hl),cosine=float(gl@hl)/max(np.linalg.norm(gl)*np.linalg.norm(hl),1e-30),linear_contribution_lr001=.01*clip*float(gl@hl)))
   if objective=="waveform":
    for lr in [.001,.01,.1]:
     old=lookup[(int(t),lr,"refitted")];assert abs(norm-old["gradient_norm"])<1e-5
     replay.append(dict(subject=sid,t=int(t),lr=lr,linear_gain=lr*clip*float(g@a["future_ce"][i].astype(float)),actual_gain=old["future_ce_gain"],current_stage=int(y[0]),transition=transition))
pd.DataFrame(rows).to_csv(o/"layer_pulses.csv",index=False);f=pd.DataFrame(full);f.to_csv(o/"full_pulses.csv",index=False);c=pd.DataFrame(coverage);c.to_csv(o/"coverage.csv",index=False);r=pd.DataFrame(replay);r.to_csv(o/"linear_vs_actual.csv",index=False)
layerperson=pd.DataFrame(rows).groupby(["subject","objective","target","layer"]).agg(dot=("dot","mean"),cosine=("cosine","mean"),contribution=("linear_contribution_lr001","mean")).reset_index();layerperson.to_csv(o/"layer_people.csv",index=False)
layers=layerperson.groupby(["objective","target","layer"]).agg(mean_contribution=("contribution","mean"),positive_people=("contribution",lambda v:int((v>0).sum())),mean_cosine=("cosine","mean"));layers.to_csv(o/"layers.csv")
overall=f.groupby(["subject","objective","target"]).agg(dot=("dot","mean"),cosine=("cosine","mean"),linear_gain=("linear_gain_lr001","mean")).reset_index();summary=overall.groupby(["objective","target"]).agg(mean_linear_gain=("linear_gain","mean"),positive_people=("linear_gain",lambda v:int((v>0).sum())),mean_cosine=("cosine","mean"));summary.to_csv(o/"summary.csv")
conditional=[]
for key in ["current_stage","transition"]:
 pp=r[r.lr==.01].groupby(["subject",key]).agg(actual_gain=("actual_gain","mean"),linear_gain=("linear_gain","mean"),pulses=("t","count")).reset_index();pp.to_csv(o/(key+"_people.csv"),index=False)
 ss=pp.groupby(key).agg(actual_gain=("actual_gain","mean"),linear_gain=("linear_gain","mean"),people=("subject","count"),pulses=("pulses","sum")).reset_index();ss["condition"]=key;ss["value"]=ss[key].astype(str);conditional.append(ss[["condition","value","actual_gain","linear_gain","people","pulses"]])
conditional=pd.concat(conditional);conditional.to_csv(o/"conditional.csv",index=False)
comparison=r.groupby(["subject","lr"])[["actual_gain","linear_gain"]].mean().groupby("lr").mean();comparison.to_csv(o/"linear_comparison.csv")
out=dict(status="PASS",pulses=256,gradient_vectors=1280,parameters_per_vector=9600,finite_difference_checks=len(fd),max_fd_absolute_error=float(fd.absolute_error.max()),diagnostic_labels_only=True,no_layer_selection_or_policy_trained=True);(o/"audit.json").write_text(json.dumps(out,indent=2))
(o/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>波形与熵目标的梯度方向诊断</h1><p>事后机制分析，固定源侧256个时点。正内积表示辅助梯度下降在局部一阶近似中降低分类损失；不等于正确率或在线 Macro-F1 提高。监督梯度只用于诊断，不用于在线更新。</p>'+summary.to_html()+'<h2>一阶预期与实际单次收益</h2>'+comparison.to_html()+'<h2>重建主条件按当前阶段与切换分组</h2><p>先按人求均值，再跨人平均；稀有阶段覆盖和样本数同时列出。阶段和切换分组使用真值，是事后分析，不是部署可用的门控特征。</p>'+conditional.to_html(index=False)+'<h2>重建目标的未来分类分层贡献</h2>'+layers.loc[("waveform","future_ce")].to_html()+'<p>分层正值不能直接证明选择这些层有效。没有训练层选择器或策略，也没有推进目标人群更新。</p><p><a href="audit.json">审计</a> · <a href="finite_differences.csv">中心差分检查</a> · <a href="layer_people.csv">逐人分层结果</a></p>')
print(summary.to_string());print(comparison.to_string());print(conditional.to_string(index=False));print(layers.loc[("waveform","future_ce")].to_string());print(out)

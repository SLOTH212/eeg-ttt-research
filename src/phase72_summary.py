from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"phase72"
df=pd.read_csv(OUT/"validation.csv");checks=[]
for sid in [5,15,25,35]:
 a=np.load(OUT/"cases"/f"s{sid:02d}.npz");old=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_bernoulli.npz");x=np.load(ROOT/"phase62/cases"/f"s{sid:02d}_existing_input.npz")["x"];mask=a["mask"]
 assert np.array_equal(mask,old["mask"]) and np.array_equal(a["masked_target"],x[mask])
 counts=mask.sum((1,2));offset=np.r_[0,counts.cumsum()];re=[]
 template=(x*(~mask)[...,None]).sum(2)/(~mask).sum(2)[...,None]
 for i,(lo,hi) in enumerate(zip(offset[:-1],offset[1:])):
  target=a["masked_target"][lo:hi].astype(float);pred=a["masked_prediction"][lo:hi].astype(float);sh=a["masked_shuffle_prediction"][lo:hi].astype(float);vis=np.broadcast_to(template[i,:,None,:],x[i].shape)[mask[i]].astype(float)
  re.append([((pred-target)**2).mean(),(target**2).mean(),((sh-target)**2).mean(),((vis-target)**2).mean()])
 re=np.array(re);err=float(abs(re-a["window_metrics"]).max());assert err<1e-12
 row=df[df.subject==sid].iloc[0];assert np.allclose(re.mean(0),row[["mse","zero_mse","shuffled_mse","visible_template_mse"]].to_numpy(float),atol=1e-12,rtol=0)
 checks.append(dict(subject=sid,max_error=err))
r=np.load(OUT/"readout.npz");scale=r["feature_scale"];A=r["covariance"]/scale[:,None]/scale[None,:]+float(r["alpha"])/int(r["total_masked_patches"])*np.eye(200);B=r["cross_covariance"]/scale[:,None];res=float(np.linalg.norm(A@r["standardized_coef"].T-B)/np.linalg.norm(B));assert res<1e-10
assert np.array_equal(r["coef"],(r["standardized_coef"]/scale[None,:]).astype(np.float32))
assert np.allclose(r["bias"],r["target_mean"]-r["standardized_coef"]/scale[None,:]@r["feature_mean"],atol=1e-7)
fit=json.loads((OUT/"fitting_complete.json").read_text());decision=json.loads((OUT/"decision.json").read_text())
assert fit["classification_labels_not_read"] and fit["nonprojection_state_exact"] and all(v["before_after_logits_exact"] for v in fit["classification_checks"])
assert decision["validity_pass"]==bool(df.validity_pass.all())
for name in ["execution_audit","regression_algebra_audit"]:assert json.loads((OUT/(name+".json")).read_text())["status"]=="PASS"
audit=dict(status="PASS",cases=checks,normal_equation_residual=res,classification_windows_exact=sum(v["windows"] for v in fit["classification_checks"]),source_windows=fit["source_windows"],extra_class_labels=0)
(OUT/"audit.json").write_text(json.dumps(audit,indent=2))
(OUT/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Phase72 波形投影校准</title><style>body{max-width:1100px;margin:40px auto;font:17px sans-serif}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>无标签源波形重新拟合重建投影</h1><p>固定分类器，只拟合原有 200→200 线性波形投影，共 40,200 参数。源侧 10 人、27,238 个窗口、816,696 个遮蔽片段；固定 Ridge α=100，按人等权。没有额外使用类别标签，但这是额外头部拟合，不能称为直接复用。</p><p>4/4 源验证者的遮蔽重建均优于零预测及输入打乱对照。分类器 10,812 个验证窗口前后 logits 完全一致。这只建立辅助重建有效性，不建立测试时分类更新收益。原始未校准头在 Phase68 的 0/4 失败结果保留。</p>'+df.to_html(index=False)+'<p>这些源验证者已反复使用，不是新盲测；只对应分类器种子 17。</p><p><a href="audit.json">独立审计</a> · <a href="protocol.json">预先固定协议</a> · <a href="fitting_complete.json">拟合记录</a></p>')
print(audit)

from common import *
import pandas as pd,hashlib
O=ROOT/"phase90";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();sol=json.loads((O/"solutions.json").read_text());stats=json.loads((O/"quadratic_coefficients.json").read_text());r=np.load(ROOT/"phase72/readout.npz");base={"weight":r["coef"].astype(float),"bias":r["bias"].astype(float)};rows=[];maxerr=0.
for which in pr["heads"]:
 if not sol[which]["feasible"]:continue
 meta={k:v.double().numpy() for k,v in torch.load(ROOT/"phase86"/which/"head.pt",map_location="cpu",weights_only=False)["head"].items()};ck=torch.load(O/(which+"_constrained_head.pt"),map_location="cpu",weights_only=False);head={k:v.double().numpy() for k,v in ck["head"].items()};beta=sol[which]["beta"]
 for k in head:assert np.allclose(head[k],base[k]+beta*(meta[k]-base[k]),atol=2e-7,rtol=0)
 for sid in pr["source_subjects"]:
  a=np.load(O/"source_cases"/f"s{sid:02d}.npz");z=a["z"].astype(float);sz=a["shuffle_z"].astype(float);y=a["target"].astype(float);p0=z@base["weight"].T+base["bias"];sh0=sz@base["weight"].T+base["bias"];pred=z@head["weight"].T+head["bias"];sh=sz@head["weight"].T+head["bias"];values=[];start=0
  for count in a["counts"]:
   sl=slice(start,start+count);values.append([np.square(p0[sl]-y[sl]).mean(),np.square(sh0[sl]-y[sl]).mean(),np.square(y[sl]).mean(),np.square(pred[sl]-y[sl]).mean(),np.square(sh[sl]-y[sl]).mean()]);start+=count
  c,sc,zero,actual,shuffle=np.mean(values,axis=0);st=next(v for v in stats if v["subject"]==sid and v["head"]==which);estimate=st["a"]*beta**2+st["b"]*beta+st["c"];err=abs(actual-estimate);maxerr=max(maxerr,err);assert err<1e-7 and abs(c-st["c"])<1e-12
  zero_margin=.5*(c+zero)-actual;shuffle_margin=(shuffle-actual)-.5*(sc-c);assert zero_margin>=-1e-7 and shuffle_margin>=-1e-7
  rows.append(dict(subject=sid,head=which,beta=beta,initial_mse=c,constrained_mse=actual,zero_mse=zero,shuffle_mse=shuffle,zero_constraint_margin=zero_margin,shuffle_constraint_margin=shuffle_margin,quadratic_actual_error=err))
df=pd.DataFrame(rows);df.to_csv(O/"source_validation.csv",index=False);audit=dict(status="PASS",source_cases=len(rows),max_quadratic_vs_float32_head_mse_error=maxerr,constraints_recomputed_from_cached_latents=True,all_initial_source_heads_valid=True,no_classification_labels_used_in_constraint=True,classifier_unchanged=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>源无标签重建约束：解析投影辅助头</h1><p>此前无约束元训练的重建失败保持不变。这是后续探索分支：在线性辅助头的初始与元训练端点间做一次共享插值，要求每位源训练者至少保留初始重建相对零输出、乱序输入的半数优势。只用原317个源片段，不使用分类指标或新标签选择插值系数。该约束比例属于预先固定的设计选择，不是经过目标结果搜索的最优值。</p><pre>'+json.dumps(sol,indent=2)+'</pre>'+df.to_html(index=False,float_format=lambda v:f"{v:.8f}")+'<p>源约束成立不代表未参与约束的人群有效，也不代表任务写入有益。下一步在原两组源评估人群固定检验，且保留先前标签成本1,268/317的披露。</p><p><a href="audit.json">独立线性与MSE复算</a> · <a href="protocol.json">固定解析规则</a></p>')
print(sol);print(audit)

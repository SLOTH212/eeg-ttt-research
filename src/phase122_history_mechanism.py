from phase119_models import *
from phase2_temporal import ema
import pandas as pd,hashlib
O=ROOT/"phase122";O.mkdir(exist_ok=True);(O/"predictions").mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=json.loads((ROOT/"phase120/audit.json").read_text())["records"];assert json.loads((ROOT/"phase120/summary_audit.json").read_text())["status"]=="PASS"
specs=[dict(seed=seed,path=str(ROOT/"phase119"/f"history11_seed{seed}"/"model.pt"),sha256=sha(ROOT/"phase119"/f"history11_seed{seed}"/"model.pt")) for seed in [17,29,43]]
pr=dict(purpose="Fixed posthoc temporalmechanism controls on cachedphase120predictions, allcohortsalreadyobserved; no newtraining/tuning.",models=specs,methods={"mean_probability11":"Uniform arithmeticmean of frozenparent probabilities overcurrentand10pastwindows, edgepadfirstwindow; exactlysame11windowextent, no training.","reverse_past10":"Existingselectedhistoryhead, reverseorder ofonly10pastlogitvectors whilekeepcurrentlastvector. Preservesmultiset, changeslagpositions; sensitivitydiagnostic, notdeployedreplacement."},evaluation="SameEMA1/.8/.5; comparepreviousselectedhistory vsnew11windowmean andorderedhistory vsreversedpast. Three-seed metricmeanswithinperson, pairedperson10000bootstrap seed122017. No primaryclaim replacesphase120.",scope="150existingcase-inputconditions. Noonlineparameterupdates or newtargetpeople; labelsusedonlyforevaluation.",cpu_ids=list(range(8,24)),gpu=False)
p=O/"protocol.json"
if p.exists():assert json.loads(p.read_text())==pr
else:p.write_text(json.dumps(pr,indent=2));(O/"protocol.sha256").write_text(sha(p))
heads={};states={}
for sp in specs:
 ck=torch.load(sp["path"],map_location="cpu",weights_only=False);m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.eval().requires_grad_(False);heads[sp["seed"]]=m;states[sp["seed"]]={k:v.clone() for k,v in m.state_dict().items()}
rows=[];checks=[]
for rec in records:
 c=rec["case"];mode=rec["mode"];name="_".join(c[k] for k in ["domain","subject","montage"]);path=ROOT/"phase120/predictions"/f"{name}_{mode}.npz";assert sha(path)==rec["prediction_sha256"];z=np.load(path);loaded=np.load(c["labels"]);y=loaded["y"] if c["labels"].endswith(".npz") else loaded;outputs={}
 for seed,m in heads.items():
  base=z[f"frozen_seed{seed}"];logits=z[f"logits_seed{seed}"];ordered=transform(logits,"history11");rev=ordered.reshape(-1,11,5).copy();rev[:,:10]=rev[:,:10][:,::-1].copy();rev=rev.reshape(-1,55);assert np.array_equal(rev[:,-5:],logits)
  short=transform(logits[:65],"history11").reshape(-1,11,5).copy();short[:,:10]=short[:,:10][:,::-1].copy();assert np.array_equal(short.reshape(-1,55),rev[:65])
  mean=context(base).reshape(-1,11,5).mean(1,dtype=np.float64).astype(np.float32);assert np.array_equal(mean[:65],context(base[:65]).reshape(-1,11,5).mean(1,dtype=np.float64).astype(np.float32))
  padded=np.pad(base.astype(float),((10,0),(0,0)),mode="edge");cum=np.vstack([np.zeros((1,5)),padded.cumsum(0)]);alt=((cum[11:]-cum[:-11])/11).astype(np.float32);me=float(abs(mean-alt).max());assert me<1e-7
  with torch.inference_mode():
   reversed_p=torch.cat([m(torch.from_numpy(rev[a:a+1024])).softmax(1) for a in range(0,len(rev),1024)]).numpy()
   replay=torch.cat([m(torch.from_numpy(ordered[a:a+1024])).softmax(1) for a in range(0,len(ordered),1024)]).numpy()
  saved=z[f"history11_seed{seed}"];err=float(abs(saved-replay).max());assert err<2e-5 and np.array_equal(saved.argmax(1),replay.argmax(1));assert all(torch.equal(v,states[seed][k]) for k,v in m.state_dict().items());checks.append(dict(case=name,mode=mode,seed=seed,cpu_gpu_head_replay_error=err,mean_independent_error=me,current_preserved_exact=True,prefix_exact=True))
  outputs[f"mean_probability11_seed{seed}"]=mean;outputs[f"reverse_past10_seed{seed}"]=reversed_p
  for variant,q in dict(mean_probability11=mean,reverse_past10=reversed_p).items():
   for alpha in [1.,.8,.5]:
    pp=q if alpha==1 else ema(q,alpha);met=metrics(y,pp);rows.append(dict(domain=c["domain"],subject=c["subject"],montage=c["montage"],mode=mode,seed=seed,variant=variant,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
 np.savez_compressed(O/"predictions"/f"{name}_{mode}.npz",**outputs)
df=pd.DataFrame(rows);assert len(df)==2700;df.to_csv(O/"new_streams.csv",index=False);old=pd.read_csv(ROOT/"phase120/participants.csv");allrows=pd.concat([df,old],ignore_index=True);keys=["domain","subject","montage","mode","variant","ema"];person=allrows.groupby(keys)[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();person.to_csv(O/"people.csv",index=False);summary=person.groupby(["domain","montage","mode","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
comparisons=[]
for (domain,montage,mode,alpha),g in person.groupby(["domain","montage","mode","ema"]):
 tab=g.pivot(index="subject",columns="variant",values="macro_f1");rng=np.random.default_rng(122017);ii=rng.integers(0,len(tab),(10000,len(tab)))
 for new,oldv in [("history11","mean_probability11"),("history11","reverse_past10"),("mean_probability11","frozen")]:
  d=tab[new]-tab[oldv];ci=np.quantile(d.to_numpy()[ii].mean(1),[.025,.975]);comparisons.append(dict(domain=domain,montage=montage,mode=mode,ema=alpha,comparison=new+"-minus-"+oldv,new_f1=float(tab[new].mean()),old_f1=float(tab[oldv].mean()),difference_pp=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_people=int((d>0).sum()),harm_over1pp=int((d < -1).sum()),worst_pp=float(d.min()),people=len(d)))
comp=pd.DataFrame(comparisons);comp.to_csv(O/"comparisons.csv",index=False);audit=dict(status="PASS",new_streams=len(df),max_cpu_gpu_replay_error=max(r["cpu_gpu_head_replay_error"] for r in checks),max_independent_mean_error=max(r["mean_independent_error"] for r in checks),prefix_and_current_preservation_exact=True,weights_unchanged_exact=True,cpu_only=True,checks=checks);(O/"audit.json").write_text(json.dumps(audit,indent=2))
show=comp[(comp.ema==.5)&comp.montage.isin(["native","left"])]
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>历史收益机制：同窗口平均与过去顺序倒置</h1><p>复用phase120完整记录预测，固定两个后续诊断，不重新训练。11窗均值用冻结父模型当前加过去10窗概率的均匀平均，首窗边缘填充；与训练头相同时间范围。倒序诊断保持当前logit不变，只反转过去十个向量的顺序，保持历史集合，但不符合原训练时的时间排列，因此仅用于敏感性解释。</p><p>全程CPU8–23；原历史头CPU预测与保存GPU预测兼容、分类一致；平均用独立累计和算法重算，前缀和当前向量保持均审计。所有目标此前已观察，这些新增对照不能替换phase120预定主比较。</p>'+show.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>三个种子在人内平均，再按人配对区间。均值对照不消耗额外标签，学习头有额外源30轮监督训练。倒置后性能变化不能直接证明真实睡眠阶段因果机制；没有据此选择窗口长度或重新优化权重。</p><p><a href="protocol.json">固定诊断协议</a> · <a href="audit.json">审计</a> · <a href="comparisons.csv">全部EMA与蒙太奇</a></p>')
print(show.to_string(index=False));print({k:v for k,v in audit.items() if k!="checks"})

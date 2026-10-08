from phase119_models import *
from phase2_temporal import ema
import pandas as pd,hashlib
os.environ["CUDA_VISIBLE_DEVICES"]="0"
O=ROOT/"phase125";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip();assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
rows=[];checks=[];sels=[]
for backbone,phase in pr["sources"].items():
 for seed in pr["seeds"]:
  for variant in pr["variants"]:
   out=O/f"{backbone}_{variant}_seed{seed}";done=json.loads((out/"complete.json").read_text());assert hashlib.sha256((out/"model.pt").read_bytes()).hexdigest()==done["checkpoint_sha256"];hist=done["history"];base=hist[0];eligible=[]
   for h in hist:
    ok=h["mean_f1"]>=base["mean_f1"]-1 and all(h["people"][str(s)]>=base["people"][str(s)]-2 for s in pr["validation"]);assert ok==h["eligible"]
    if ok:eligible.append(h)
   best=max(eligible,key=lambda h:h["mean_f1"]);assert best["epoch"]==done["selected_epoch"];sels.append(dict(backbone=backbone,seed=seed,variant=variant,epoch=done["selected_epoch"],macro_f1=done["selected_f1"]))
   for stage,file in [("selected","model.pt"),("final30","final30.pt")]:
    ck=torch.load(out/file,map_location="cpu",weights_only=False);m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for sid in pr["validation"]:
     data=np.load(ROOT/phase/"logits"/f"seed{seed}_s{sid:02d}.npz");y=data["y"];x=torch.from_numpy(transform(data["logits"],variant)).cuda()
     with torch.inference_mode():p=torch.cat([m(x[a:a+1024]).softmax(1).cpu() for a in range(0,len(x),1024)]).numpy()
     score=100*metrics(y,p)["macro_f1"];err=abs(score-hist[ck["epoch"]]["people"][str(sid)]);assert err<1e-10
     if stage=="selected":assert np.array_equal(p,np.load(out/f"s{sid:02d}.npz")["p"])
     checks.append(dict(backbone=backbone,seed=seed,variant=variant,stage=stage,subject=sid,score_error=err));candidates={stage+"_"+variant:p}
     if variant=="repeat11" and stage=="selected":
      parent=torch.from_numpy(data["logits"]).softmax(1).numpy();teacher=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"context11_seed{seed}"];candidates.update(frozen=parent,teacher=teacher,static_half_blend=.5*parent+.5*teacher)
      for v in ["repeat11","history11"]:candidates["original_"+v]=np.load(ROOT/phase/f"{v}_seed{seed}"/f"s{sid:02d}.npz")["p"]
     for name,q in candidates.items():
      for alpha in [1.,.8,.5]:
       pp=q if alpha==1 else ema(q,alpha);met=metrics(y,pp);rows.append(dict(backbone=backbone,seed=seed,subject=sid,variant=name,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
    assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
df=pd.DataFrame(rows);assert len(df)==648;df.to_csv(O/"streams.csv",index=False);person=df.groupby(["backbone","subject","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();person.to_csv(O/"people.csv",index=False);summary=person.groupby(["backbone","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False)
decisions=[]
for backbone in pr["sources"]:
 tab=person[(person.backbone==backbone)&(person.ema==1)].pivot(index="subject",columns="variant",values="macro_f1");delta=tab.selected_history11-tab.original_history11;repeat=tab.selected_history11-tab.selected_repeat11;advance=bool(delta.mean()>0 and (delta>0).sum()>=3 and repeat.mean()>0)
 decisions.append(dict(backbone=backbone,history_vs_original_delta=float(delta.mean()),positive_people=int((delta>0).sum()),history_vs_weighted_repeat=float(repeat.mean()),advance_targets=advance))
(O/"decision.json").write_text(json.dumps(decisions,indent=2));audit=dict(status="PASS",reconstructed_cases=len(checks),streams=len(df),max_score_replay_error=max(r["score_error"] for r in checks),selected_probabilities_exact=True,guards_recomputed=True,no_targets_evaluated=True);(O/"source_replay_audit.json").write_text(json.dumps(audit,indent=2));pd.DataFrame(sels).to_csv(O/"selections.csv",index=False)
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>源阶段内稳定/转换权重平衡</h1><p>两种冻结基础模型、三个种子、同容量重复与历史头，共12个训练分支。原始源10人的同一27227个已评分窗口、30轮、同初始化和顺序；只在每个真实当前阶段内，使已知11窗的稳定/转换部分拥有相等总训练权重。启动或历史未知权重因子为1。权重由源标签计算，不匹配目标阶段比例；标签没有进入测试特征或门控。</p><p>原始源保护门槛和epoch0回退保留。推进规则要求人内三种子平均后，新的历史头优于原历史头、至少3/4人改善且优于新的重复头；未通过不进入目标集或放宽权重。</p>'+pd.DataFrame(decisions).to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>源raw结果</h2>'+summary[summary.ema==1].to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>轮次</h2>'+pd.DataFrame(sels).to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<p>这只检验一个固定的源权重策略；负结果不证明时间结构差异没有影响，也不能把最终轮次或次要EMA替换主选择。<a href="weight_audit.json">权重质量守恒审计</a> · <a href="source_replay_audit.json">检查点独立重放</a> · <a href="protocol.json">协议</a></p>')
print(pd.DataFrame(decisions).to_string(index=False));print(audit)

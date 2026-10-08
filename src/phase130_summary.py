from phase119_models import *
from phase2_temporal import ema
import pandas as pd,hashlib
os.environ["CUDA_VISIBLE_DEVICES"]="0"
O=ROOT/"phase130";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip();assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
rows=[];checks=[];selections=[];trajectory=[]
for backbone,phase in pr["backbone_phases"].items():
 for seed in pr["seeds"]:
  for variant in pr["variants"]:
   out=O/f"{backbone}_{variant}_seed{seed}";done=json.loads((out/"complete.json").read_text());hist=done["history"];base=hist[0];eligible=[]
   for h in hist:
    ok=h["mean_f1"]>=base["mean_f1"]-1 and all(h["people"][str(s)]>=base["people"][str(s)]-2 for s in pr["new_selector"]);assert ok==h["eligible"]
    if ok:eligible.append(h)
   best=max(eligible,key=lambda h:h["mean_f1"]);assert best["epoch"]==done["selected_epoch"];selections.append(dict(backbone=backbone,seed=seed,variant=variant,old_epoch=done["old_selected_epoch"],new_epoch=done["selected_epoch"],new_selector_f1=done["selected_f1"],old_selector_f1=float(np.mean(list(done["original_selector_evaluation"].values())))))
   trajectory.append(dict(backbone=backbone,seed=seed,variant=variant,selected_state_error=done["old_selected_reproduction"]["parameter_error"],final_state_error=done["old_final30_parameter_error"],max_epoch_metric_error=done["max_old_selector_epoch_macro_f1_error"]))
   for selector,folder in [("original",ROOT/phase/f"{variant}_seed{seed}"),("alternate",out)]:
    path=folder/"model.pt";ck=torch.load(path,map_location="cpu",weights_only=False);expected=json.loads((folder/"complete.json").read_text())["checkpoint_sha256"];assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for group,sids in [("original_selector",pr["original_selector"]),("alternate_selector",pr["new_selector"])]:
     for sid in sids:
      inputpath=ROOT/phase/"logits"/f"seed{seed}_s{sid:02d}.npz" if group=="original_selector" else O/"controller_logits"/f"{backbone}_seed{seed}_s{sid:02d}.npz";d=np.load(inputpath);xx=torch.from_numpy(transform(d["logits"],variant)).cuda();y=d["y"]
      with torch.inference_mode():p=torch.cat([m(xx[a:a+1024]).softmax(1).cpu() for a in range(0,len(xx),1024)]).numpy()
      if selector=="alternate":
       gold=np.load(out/f"s{sid:02d}.npz")["p"];assert np.array_equal(p,gold)
       if group=="alternate_selector":assert abs(100*metrics(y,p)["macro_f1"]-done["people"][str(sid)])<1e-10
      checks.append(dict(backbone=backbone,seed=seed,variant=variant,selector=selector,group=group,subject=sid))
      candidates={(selector,variant):p}
      if selector=="original" and variant=="current":candidates[("parent","frozen")]=torch.from_numpy(d["logits"]).softmax(1).numpy()
      for (sel,v),q in candidates.items():
       for alpha in [1.,.8,.5]:
        qq=q if alpha==1 else ema(q,alpha);met=metrics(y,qq);rows.append(dict(backbone=backbone,seed=seed,subject=sid,group=group,selector=sel,variant=v,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
    assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
df=pd.DataFrame(rows);assert len(df)==1008;df.to_csv(O/"source_streams.csv",index=False);person=df.groupby(["backbone","subject","group","selector","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();person.to_csv(O/"source_people.csv",index=False);summary=person.groupby(["backbone","group","selector","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"source_summary.csv",index=False);pd.DataFrame(selections).to_csv(O/"selections.csv",index=False);pd.DataFrame(trajectory).to_csv(O/"trajectory_replay.csv",index=False)
audit=dict(status="PASS",condition_subject_replays=len(checks),streams=len(df),new_selected_predictions_bitwise_exact=True,new_selector_guards_recomputed=True,max_old_selected_parameter_error=max(r["selected_state_error"] for r in trajectory),max_old_final_parameter_error=max(r["final_state_error"] for r in trajectory),max_old_epoch_metric_error=max(r["max_epoch_metric_error"] for r in trajectory),new_source_only_selection=True,no_target_evaluation_yet=True)
(O/"source_replay_audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1400px;margin:35px auto}td,th{padding:5px;border:1px solid #ddd}</style><h1>源选模人群替换：静态历史方法的敏感性</h1><p>源微调父模型和头训练仍使用原10人，30轮训练配方、初始参数、批次顺序与dropout随机序列保持不变。把选模4人从5/15/25/35替换为7/17/27/37，两组均不参与父模型源微调或头训练。所有人此前已被研究观察，不是新的盲验证。</p><p>重建未保存的训练中间状态，并在原所选轮次及第30轮逐元素对照历史检查点；本轮额外复算开销属于重现工作。新轮次仅由替换后的四位源受试者和同一原始输入保护门槛决定，不根据目标表现挑选验证组。</p>'+pd.DataFrame(selections).to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>父模型检查点仍由原源验证组选择；本轮只替换读出头的选模组，不是整个训练流程的交叉验证。</p><h2>两组源数据的交叉评价，raw概率</h2>'+summary[summary.ema==1].to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>独立重放</h2><pre>'+json.dumps(audit,indent=2)+'</pre><p>同组分数参与了该组策略的选模，不能当独立泛化结果。后续将固定评价所有新选择及回退模型，不只报告更有利的验证组。<a href="protocol.json">固定敏感性协议</a> · <a href="trajectory_replay.csv">逐模型训练重现</a></p>')
print(summary[summary.ema==1].to_string(index=False));print(audit)

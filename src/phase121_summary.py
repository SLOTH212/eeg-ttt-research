from phase119_models import *
from phase2_temporal import ema
import pandas as pd,hashlib
os.environ["CUDA_VISIBLE_DEVICES"]="0"
O=ROOT/"phase121";C=O/"logits";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text().strip();assert json.loads((O/"audit.json").read_text())["status"]=="PASS"
rows=[];audits=[];selections=[]
for seed in pr["seeds"]:
 for variant in pr["variants"]:
  out=O/f"{variant}_seed{seed}";done=json.loads((out/"complete.json").read_text());assert hashlib.sha256((out/"model.pt").read_bytes()).hexdigest()==done["checkpoint_sha256"];hist=done["history"];base={int(k):v for k,v in hist[0]["people"].items()}
  eligible=[]
  for h in hist:
   ok=h["mean_f1"]>=hist[0]["mean_f1"]-1 and all(float(h["people"][str(s)])>=base[s]-2 for s in pr["validation"]);assert ok==h["eligible"]
   if ok:eligible.append(h)
  best=max(eligible,key=lambda h:h["mean_f1"]);assert best["epoch"]==done["selected_epoch"];selections.append(dict(seed=seed,variant=variant,selected_epoch=done["selected_epoch"],macro_f1=done["selected_f1"],parent_f1=done["parent_f1"],parameters=done["parameters"]))
  for stage,file in [("selected","model.pt"),("final30","final30.pt")]:
   ck=torch.load(out/file,map_location="cpu",weights_only=False);m=ResidualLogitHead(ck["dim"]);m.load_state_dict(ck["state"]);m.cuda().eval();state={k:v.cpu().clone() for k,v in m.state_dict().items()}
   for sid in pr["validation"]:
    d=np.load(C/f"seed{seed}_s{sid:02d}.npz");ff=transform(d["logits"],variant);assert np.array_equal(ff[:65],transform(d["logits"][:65],variant));x=torch.from_numpy(ff).cuda();y=d["y"]
    with torch.inference_mode():
     p=torch.cat([m(x[a:a+1024]).softmax(1).cpu() for a in range(0,len(x),1024)]).numpy();prefix=m(x[:65]).softmax(1).cpu().numpy()
    err=float(abs(p[:65]-prefix).max());assert err<2e-5 and np.array_equal(p[:65].argmax(1),prefix.argmax(1));score=100*metrics(y,p)["macro_f1"];expected=hist[ck["epoch"]]["people"][str(sid)];assert abs(score-expected)<1e-10
    if stage=="selected":
     saved=np.load(out/f"s{sid:02d}.npz")["p"];assert np.array_equal(p,saved)
    audits.append(dict(seed=seed,variant=variant,stage=stage,subject=sid,score_error=abs(score-expected),prefix_probability_error=err))
    candidates={stage+"_"+variant:p}
    if variant=="current" and stage=="selected":
     parent=torch.from_numpy(d["logits"]).softmax(1).numpy();teacher=np.load(ROOT/"phase79/predictions"/f"source_s{sid:02d}_native_raw.npz")[f"context11_seed{seed}"];candidates.update(frozen=parent,teacher=teacher,static_half_blend=.5*parent+.5*teacher)
    for name,q in candidates.items():
     for alpha in [1.,.8,.5]:
      pp=q if alpha==1 else ema(q,alpha);met=metrics(y,pp);rows.append(dict(seed=seed,subject=sid,variant=name,ema=alpha,macro_f1=100*met["macro_f1"],balanced_accuracy=100*met["balanced_accuracy"],ce=met["ce"]))
   assert all(torch.equal(v.cpu(),state[k]) for k,v in m.state_dict().items())
df=pd.DataFrame(rows);assert len(df)==324;df.to_csv(O/"source_streams.csv",index=False);person=df.groupby(["subject","variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();person.to_csv(O/"people.csv",index=False);summary=person.groupby(["variant","ema"])[["macro_f1","balanced_accuracy","ce"]].mean().reset_index();summary.to_csv(O/"summary.csv",index=False);sel=pd.DataFrame(selections);sel.to_csv(O/"selections.csv",index=False)
table=person[person.ema==1].pivot(index="subject",columns="variant",values="macro_f1");delta=table.selected_history11-table.selected_repeat11;froz=table.selected_history11-table.frozen
advance=bool(delta.mean()>0 and froz.mean()>0 and (delta>0).sum()>=3)
decision=dict(advance_existing_target_exploratory=advance,history_repeat_mean_delta=float(delta.mean()),positive_people=int((delta>0).sum()),history_frozen_mean_delta=float(froz.mean()),person_history_repeat_delta={int(s):float(v) for s,v in delta.items()},source_previously_observed=True,no_online_adaptation=True)
(O/"decision.json").write_text(json.dumps(decision,indent=2));(O/"source_replay_audit.json").write_text(json.dumps(dict(status="PASS",replayed=len(audits),max_score_error=max(r["score_error"] for r in audits),max_prefix_probability_error=max(r["prefix_probability_error"] for r in audits),selected_probabilities_bitwise_exact=True,selection_guards_independently_recomputed=True,head_states_frozen_exact=True,records=audits),indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>冻结LaBraM的静态因果历史残差头</h1><p>冻结三个原始LaBraM源模型；取每窗5类logit并减去类均值，输入当前5维、重复当前11次55维、或过去10窗加当前55维。残差MLP输出加回当前logit，输出层初始为零，因此起点等同冻结模型。历史与重复分支参数量、初始参数、30轮训练顺序均一致。</p><p>原10位源训练者/4位源验证者，额外监督训练预算明确计入。原始概率Macro-F1选模，均值/逐人保护门槛事先固定，epoch0回退；EMA仅次要评分。数据已观察，不是新盲验证，不属于TTT。</p>'+sel.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>人内三种子均值</h2>'+summary.to_html(index=False,float_format=lambda v:f"{v:.7f}")+'<h2>预定推进条件</h2><pre>'+json.dumps(decision,indent=2)+'</pre><p><a href="protocol.json">固定协议</a> · <a href="source_replay_audit.json">独立重放与前缀审计</a></p>')
print(sel.to_string(index=False));print(summary.to_string(index=False));print(decision)

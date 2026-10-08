from phase96_full_amplitude import *
import pandas as pd
def run(seed):
 global O
 assert seed in [29,43]
 O=ROOT/"phase96"/f"seed{seed}"
 assert json.loads((O/"complete.json").read_text())["status"]=="PASS";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();sel=json.loads((O/"selection.json").read_text());records={m:json.loads((O/m/"complete.json").read_text()) for m in pr["methods"]};hist={m:json.loads((O/m/"history.json").read_text()) for m in pr["methods"]};epochs=[]
 assert records["repeat_raw"]["label_presentations"]==records["amplitude_augmented"]["label_presentations"]
 for method in pr["methods"]:
  history=hist[method]["history"];assert len(history)==pr["epochs"]+1;initial=history[0];native={v["subject"]:v["macro_f1"] for v in initial["person_scores"] if v["gain"]==1};best=initial["robust_f1"];be=0
  gen=torch.Generator(device="cpu").manual_seed(pr["classifier_seed"]);orders=[hashlib.sha256(torch.randperm(records[method]["training_windows"]*3,generator=gen).numpy().tobytes()).hexdigest() for _ in range(pr["epochs"])];assert orders==hist[method]["permutation_hashes"]
  for row in history:
   score=np.mean([v["macro_f1"] for v in row["person_scores"]]);ns=[v for v in row["person_scores"] if v["gain"]==1];nm=np.mean([v["macro_f1"] for v in ns]);worst=min(v["macro_f1"]-native[v["subject"]] for v in ns);eligible=nm>=initial["native_f1"]-1 and worst>=-2
   assert abs(score-row["robust_f1"])<1e-10 and abs(nm-row["native_f1"])<1e-10 and eligible==row["eligible"]
   if eligible and score>best:best=score;be=row["epoch"]
   epochs.append(dict(method=method,epoch=row["epoch"],robust_f1=score,native_f1=nm,native_delta_pp=nm-initial["native_f1"],worst_native_delta_pp=worst,eligible=eligible))
  assert be==sel["selection"][method]["epoch"] and abs(best-sel["selection"][method]["robust_f1"])<1e-10
 states={};aliases={};heads={}
 def fingerprint(state):
  h=hashlib.sha256()
  for key,v in sorted(state.items()):h.update(key.encode());h.update(v.contiguous().numpy().tobytes())
  return h.hexdigest()
 def add(name,state):
  key=fingerprint(state);states.setdefault(key,state);aliases[name]=key
 base=torch.load(pr["checkpoint"],map_location="cpu",weights_only=False)["state"];add("initial",base)
 for method in pr["methods"]:
  path=O/method/"model.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==sel["selection"][method]["checkpoint_sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);add(method+"_selected",ck["state"])
  final=torch.load(O/method/"resume.pt",map_location="cpu",weights_only=False);assert final["epoch"]==5;add(method+"_final_diagnostic",final["state"])
 for state in states.values():
  for key in base:
   if key.startswith("encoder.proj_out.") or key in ["mean","scale"]:assert torch.equal(state[key],base[key])
 m=Carrier("pretrained").cuda().eval().requires_grad_(False);rows=[];stages=[];maxerr=0.;(O/"validation_predictions").mkdir(exist_ok=True)
 for sid in pr["validation"]:
  x,y=prepare_views(sid,pr["gains"]);cache={}
  for key,state in states.items():
   m.load_state_dict(state)
   for j,gain in enumerate(pr["gains"]):cache[key,j]=probabilities(m,x[:,j])
  for method,key in aliases.items():
   pp=np.stack([cache[key,j] for j in range(3)],axis=1);np.savez_compressed(O/"validation_predictions"/f"{method}_s{sid:02d}.npz",p=pp,y=y,gains=np.array(pr["gains"]))
   for j,gain in enumerate(pr["gains"]):
    mm=metrics(y,pp[:,j]);cm=np.array(mm["confusion"]);f1=100*mm["macro_f1"];rows.append(dict(method=method,subject=sid,gain=gain,macro_f1=f1,accuracy=100*cm.trace()/cm.sum(),ce=mm["ce"]))
    if method.endswith("_selected"):
     tag=method.removesuffix("_selected");expected=next(v["macro_f1"] for v in sel["selection"][tag]["selected_metrics"]["person_scores"] if v["subject"]==sid and v["gain"]==gain);err=abs(f1-expected);maxerr=max(maxerr,err);assert err<1e-8
    for k,stage in enumerate(["W","N1","N2","N3","REM"]):stages.append(dict(method=method,subject=sid,gain=gain,stage=stage,f1=200*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1),recall=100*cm[k,k]/max(cm[k].sum(),1),true_windows=int(cm[k].sum()),predicted_windows=int(cm[:,k].sum())))
  print("CHECKPOINT REPLAY",sid,len(states),flush=True)
 df=pd.DataFrame(rows);df.to_csv(O/"validation_participants.csv",index=False);pd.DataFrame(stages).to_csv(O/"stage_participants.csv",index=False);pd.DataFrame(epochs).to_csv(O/"epoch_tradeoff.csv",index=False);summary=df.groupby(["method","gain"])[["macro_f1","accuracy","ce"]].mean();summary.to_csv(O/"validation_summary.csv")
 advance=sel["selection"]["amplitude_augmented"]["epoch"]>0 and sel["selection"]["amplitude_augmented"]["robust_f1"]>sel["selection"]["repeat_raw"]["robust_f1"];assert advance==sel["advance"];audit=dict(status="PASS",source_selection_independently_replayed=True,selected_checkpoint_max_f1_error=maxerr,source_validation_prediction_streams=len(df),unique_model_states=len(states),unused_projection_and_scalers_exact=True,same_training_orders_independently_replayed=True,label_presentations_per_method=records["repeat_raw"]["label_presentations"],no_new_unique_labels=True,advance=advance,final_checkpoints_diagnostic_only=True);(O/"audit.json").write_text(json.dumps(audit,indent=2))
 (O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1450px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>完整CBraMod编码器：额外训练种子的源幅值增强复现</h1><p>原始输入重复三次与0.1/1/10倍幅值三个视图采用相同训练者、标签呈现、参数范围、批次顺序和五轮预算。只使用源标签；原幅值均值下降≤1pp、每人下降≤2pp的保护条件保持不变，在满足条件的检查点中按源三幅值平均Macro-F1选优。方法受到此前外部幅值问题启发，后续外部评估只能称为探索性。</p><h2>源选择</h2><pre>'+json.dumps(sel,indent=2)+'</pre><h2>完整训练曲线</h2>'+pd.DataFrame(epochs).to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>独立检查点复算</h2>'+summary.reset_index().to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p>final_diagnostic只用于解释固定末轮行为，不替换选择规则。static模型在记录开始前固定，推理时不更新参数。源验证保护条件不能保证外部收益或个体安全。</p><p><a href="audit.json">源选择与实际检查点审计</a> · <a href="stage_participants.csv">分期级结果</a> · <a href="protocol.json">固定方法</a> · <a href="../../phase93/report.html">保留冻结编码器失败</a></p>')
 print(sel);print(audit)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,choices=[29,43],required=True);args=parser.parse_args();run(args.seed)

from common import *
import pandas as pd,hashlib
O=ROOT/"phase93";pr=json.loads((O/"protocol.json").read_text());assert hashlib.sha256((O/"protocol.json").read_bytes()).hexdigest()==(O/"protocol.sha256").read_text();assert json.loads((O/"complete.json").read_text())["status"]=="PASS";sel=json.loads((O/"selection.json").read_text());source=json.loads((O/"cache_audit.json").read_text());assert source["status"]=="PASS";assert len(source["records"])==14
for r in source["records"]:assert hashlib.sha256((O/"cache"/f's{r["subject"]:02d}.npz').read_bytes()).hexdigest()==r["cache_sha256"] and r["encoder_state_exact"]
hist={m:json.loads((O/m/"history.json").read_text()) for m in pr["methods"]};assert hist["repeat_raw"]["permutation_hashes"]==hist["amplitude_augmented"]["permutation_hashes"];assert hist["repeat_raw"]["label_presentations"]==hist["amplitude_augmented"]["label_presentations"];rows=[];selected=[];maxscoreerr=0.
for method,h in hist.items():
 base=h["history"][0];native={r["subject"]:r["macro_f1"] for r in base["person_scores"] if r["gain"]==1};best=base["robust_f1"];epoch=0
 for r in h["history"]:
  robust=np.mean([v["macro_f1"] for v in r["person_scores"]]);nat=[v for v in r["person_scores"] if v["gain"]==1];nm=np.mean([v["macro_f1"] for v in nat]);minimum=min(v["macro_f1"]-native[v["subject"]] for v in nat);eligible=nm>=base["native_f1"]-1 and minimum>=-2;assert bool(eligible)==r["eligible"] and abs(robust-r["robust_f1"])<1e-10 and abs(nm-r["native_f1"])<1e-10
  if eligible and robust>best:best=robust;epoch=r["epoch"]
  rows.append(dict(method=method,epoch=r["epoch"],robust_f1=robust,native_f1=nm,native_mean_delta_pp=nm-base["native_f1"],worst_native_delta_pp=minimum,eligible=eligible))
 assert epoch==sel["selection"][method]["epoch"] and abs(best-sel["selection"][method]["robust_f1"])<1e-10
 path=O/method/"head.pt";assert hashlib.sha256(path.read_bytes()).hexdigest()==sel["selection"][method]["checkpoint_sha256"];ck=torch.load(path,map_location="cpu",weights_only=False);head=nn.Linear(400,5);head.load_state_dict(ck["head"]);head.eval()
 for sid in pr["validation"]:
  a=np.load(O/"cache"/f"s{sid:02d}.npz")
  with torch.no_grad():p=head(torch.tensor(a["z"].reshape(-1,400))).softmax(1).numpy().reshape(len(a["y"]),3,5)
  for j,gain in enumerate(pr["gains"]):
   f1=100*metrics(a["y"],p[:,j])["macro_f1"];expected=next(v["macro_f1"] for v in h["history"][epoch]["person_scores"] if v["subject"]==sid and v["gain"]==gain);err=abs(f1-expected);maxscoreerr=max(maxscoreerr,err);assert err<1e-8;selected.append(dict(method=method,subject=sid,gain=gain,macro_f1=f1,selected_epoch=epoch))
df=pd.DataFrame(rows);df.to_csv(O/"epoch_tradeoff.csv",index=False);pd.DataFrame(selected).to_csv(O/"selected_participants.csv",index=False);audit=dict(status="PASS",cache_hashes_verified=14,source_checkpoint_verified=hashlib.sha256(Path(pr["checkpoint"]).read_bytes()).hexdigest()==pr["checkpoint_sha256"],selected_head_cpu_replay_max_f1_error=maxscoreerr,epoch_selection_independently_replayed=True,identical_training_permutations=True,identical_label_presentations=hist["repeat_raw"]["label_presentations"],no_new_unique_labels=True,advance=sel["advance"]);assert audit["source_checkpoint_verified"];(O/"audit.json").write_text(json.dumps(audit,indent=2))
(O/"report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 sans-serif;max-width:1350px;margin:35px auto}td,th{padding:6px;border:1px solid #ddd}</style><h1>源幅值增强：冻结编码器的分类头对照</h1><p>同一10位源训练者、同一预训练后微调的CBraMod seed17编码器。两个分支仅训练2,005参数分类头，原始输入重复三次与0.1/1/10倍共同增益三个视图具有相同顺序和标签呈现次数；无新增独立标签。增益选择受到此前外部幅值问题的启发，因此方法设计并非未见外部数据的盲测。</p><p>训练前固定选择规则：在源验证三种增益的平均Macro-F1上选优，但原幅值均值下降不得超过1pp，任何一位源验证者不得下降超过2pp；包含epoch0。所有训练后增强检查点未满足该规则，因此选回初始头，不推进外部评估。保护阈值是操作性设计选择，不是统计显著性或普遍安全保证；这里的失败也不意味着幅值增强完全无效。</p>'+df.to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<h2>选定模型逐人结果</h2>'+pd.DataFrame(selected).to_html(index=False,float_format=lambda v:f"{v:.6f}")+'<p><a href="audit.json">独立选择与CPU复算</a> · <a href="protocol.json">预定方法与约束</a> · <a href="selection.json">源选择记录</a></p>')
print(df.to_string(index=False));print(audit)

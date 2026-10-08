from common import *
import ast,types,time
OUT80=ROOT/"phase80"
def entropy_logits(x):return -(x.softmax(1)*x.log_softmax(1)).sum(1)
class Algorithm(nn.Module):
 def __init__(self,input_shape,num_classes,num_domains,hparams):super().__init__();self.hparams=hparams
def official_class():
 source=(OUT80/"upstream/domainbed_adapt_algorithms.py").read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="T3A");env=dict(Algorithm=Algorithm,torch=torch,softmax_entropy=entropy_logits);exec(compile(ast.Module(body=[node],type_ignores=[]),"<pinned official T3A>","exec"),env);return env["T3A"]
def weights(supports,labels):return nn.functional.normalize(nn.functional.normalize(supports,dim=1).T@labels,dim=0)
def replay(z,head,source_p,K=20,reference_class=None):
 supports=head.weight.detach().clone();lp=head(supports);labels=nn.functional.one_hot(lp.argmax(1),5).float();ent=entropy_logits(lp);ids=torch.arange(-len(supports),0);warm_counts=labels.sum(0).int().tolist();fixedweights=weights(supports,labels);fixed=(z@fixedweights).softmax(1).numpy();fixed[0]=source_p[0];pred=[];trace=[];maxreferr=0.;initial={n:v.clone() for n,v in head.state_dict().items()}
 reference=reference_class((z.shape[1],),5,1,dict(filter_K=K,cached_loader=True),types.SimpleNamespace(featurizer=nn.Identity(),classifier=head)) if reference_class else None
 for t,row in enumerate(z):
  prior=source_p[t] if t==0 else (row[None]@weights(supports,labels)).softmax(1).numpy()[0];pred.append(prior.copy())
  logits=head(row[None]);label=nn.functional.one_hot(logits.argmax(1),5).float();supports=torch.cat([supports,row[None]]);labels=torch.cat([labels,label]);ent=torch.cat([ent,entropy_logits(logits)]);ids=torch.cat([ids,torch.tensor([t])]);classes=labels.argmax(1);indices=torch.cat([torch.where(classes==c)[0][torch.sort(ent[classes==c]).indices[:K]] for c in range(5)]);supports=supports[indices];labels=labels[indices];ent=ent[indices];ids=ids[indices];counts=labels.sum(0).int();assert int(counts.max())<=K and len(supports)<=5*K;trace.append([len(supports),int(bool((ids==t).any())),*counts.tolist()])
  if reference is not None and t<64:
   actual=reference(row[None],adapt=True);expected=row[None]@weights(supports,labels);err=float(abs(actual-expected).max());maxreferr=max(maxreferr,err);assert err<2e-5;assert torch.equal(reference.supports,supports) and torch.equal(reference.labels,labels) and torch.equal(reference.ent,ent)
 assert all(torch.equal(v,initial[n]) for n,v in head.state_dict().items());return dict(t3a20_causal=np.array(pred),source_prototype_static=fixed,trace=np.array(trace),final_supports=supports.numpy(),final_labels=labels.numpy(),final_entropy=ent.numpy(),final_ids=ids.numpy(),warm_counts=warm_counts,official_postupdate_error=maxreferr)
def run():
 pr=json.loads((OUT80/"protocol.json").read_text());assert json.loads((OUT80/"cache_audit.json").read_text())["status"]=="PASS";official=official_class();out=OUT80/"predictions";out.mkdir(exist_ok=True);checks=[]
 with torch.inference_mode():
  for spec in pr["models"]:
   key=f'{spec["model"]}_seed{spec["seed"]}';h=np.load(OUT80/"cache"/(key+"_head.npz"));head=nn.Linear(h["weight"].shape[1],5);head.weight.copy_(torch.from_numpy(h["weight"]));head.bias.copy_(torch.from_numpy(h["bias"]));head.eval().requires_grad_(False)
   for sid in pr["subjects"]:
    for mode in pr["input_modes"]:
     if spec["model"]=="relative" and mode!="raw":continue
     p=OUT80/"cache"/f"{key}_s{sid:02d}_{mode}.npz";a=np.load(p);z=torch.from_numpy(a["z"]);start=time.perf_counter();result=replay(z,head,a["p"],pr["filter_K"],official);prefix=replay(z[:65],head,a["p"][:65],pr["filter_K"]);assert np.array_equal(prefix["t3a20_causal"],result["t3a20_causal"][:65]);assert np.array_equal(result["t3a20_causal"][0],a["p"][0]);warm_counts=result.pop("warm_counts");err=result.pop("official_postupdate_error");np.savez_compressed(out/f"{key}_s{sid:02d}_{mode}.npz",frozen=a["p"],**result);checks.append(dict(model=key,subject=sid,input_mode=mode,windows=len(z),warm_class_counts=warm_counts,official_postupdate_error=err,prefix_exact=True,first_prediction_source_exact=True,head_parameters_unchanged=True,prototype_replay_seconds=time.perf_counter()-start));print("CAUSAL T3A SOURCE",key,sid,mode,"warm",warm_counts,flush=True)
 (OUT80/"execution_audit.json").write_text(json.dumps(dict(status="PASS",cases=checks,upstream_update_math_verified=True,official_current_update_prediction_only_used_for_audit=True,causal_first_source_prediction=True,bounded_supports_per_class=20,no_gradient_updates=True),indent=2));(OUT80/"complete.json").write_text(json.dumps(dict(complete=True,source_trajectories=len(checks))))
if __name__=="__main__":run()

from adaptation import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args();path=Path(args.model);out=path.parent;m,ck=load_model(path);kind=ck['kind'];sid=CFG['folds'][ck['fold']]['base_val'][0];x,y=load_subject(sid,kind);lr=json.loads((out/'lr_sensitivity.json').read_text())['selected_lr'];a=Adapter(m,lr)
changed=0;deltas=[];pred_changed=0;start=time.perf_counter()
for t in range(len(x)):
 old=a.snapshot();p,c,s=a.candidate(torch.tensor(np.array(x[t:t+1]),dtype=torch.float32));delta=float(sum((u-v).square().sum() for u,v in zip(old,c)).sqrt());deltas.append(delta);changed+=delta>0;pred_changed+=s[2]>0;a.restore(c)
summary={'source_subject':sid,'selected_lr':lr,'windows':len(x),'effective_parameter_update_fraction':changed/len(x),'prediction_change_fraction':pred_changed/len(x),'delta_norm_p50':float(np.quantile(deltas,.5)),'delta_norm_p95':float(np.quantile(deltas,.95)),'all_parameters_on_cpu':all(p.device.type=='cpu' for p in a.model.parameters()),'all_writable_fp32':all(p.dtype==torch.float32 for p in a.params),'seconds':time.perf_counter()-start,'writable_scope':ck.get('writable_scope','affine')}
assert summary['all_parameters_on_cpu'] and summary['all_writable_fp32'];(out/'numerical_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

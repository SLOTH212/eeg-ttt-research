from adaptation import *
import inspect
m,ck=load_model(ROOT/'results/spectral_A_17/model.pt');x,y=load_subject(12,'spectral');x=x[:64];y=y[:64]
hf=np.load(ROOT/'data/processed/s12_meta.npz')['features'][:64]
p,ds,tm=stream(m,x,'continuous',.001,history_features=hf,collect=True)
q,_,_=stream(m,x,'continuous',.001,history_features=hf,collect=False)
audit={'diagnostics_do_not_change_trajectory':bool(np.array_equal(p,q)), 'stream_has_no_labels_parameter':'y' not in inspect.signature(stream).parameters}
# Directly replay a sampled snapshot and independently recompute branch probabilities.
a=Adapter(m,.001)
for t in range(21):
 prob,candidate,s=a.candidate(torch.from_numpy(np.array(x[t:t+1])))
 if t==20:
  old=a.snapshot();future=torch.from_numpy(np.array(x[t+1:t+11]));d=next(d for d in ds if d['t']==20)
  with torch.no_grad():
   audit['keep_branch_matches_replay']=bool(np.array_equal(a.model(future).softmax(1).numpy(),d['keep']))
   a.restore(candidate);audit['write_branch_matches_replay']=bool(np.array_equal(a.model(future).softmax(1).numpy(),d['write']))
   a.restore(a.source);audit['reset_branch_matches_replay']=bool(np.array_equal(a.model(future).softmax(1).numpy(),d['reset']))
  a.restore(old)
 a.restore(candidate)
# Diagnostics gain sign and target indexing independently checked.
rows=score_diagnostics(ds,y,12,'continuous');d=ds[0];yy=y[1:5];good=yy>=0
expected=np.mean(np.log(d['write'][:4][good,yy[good]])-np.log(d['keep'][:4][good,yy[good]]))
reported=next(r['write_gain'] for r in rows if r['t']==0 and r['h']==4)
audit['future_gain_sign_and_index_correct']=bool(np.isclose(expected,reported,atol=1e-7))
for name,fn in [('t3a',t3a),('causal_bn',causal_bn)]:
 a=fn(m,x);b=fn(m,x[:20]);audit[name+'_prefix_causal']=bool(np.allclose(a[:20],b,atol=1e-6))
(ROOT/'results/real_data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));assert all(audit.values())

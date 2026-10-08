from phase30_masked_graph import *
from phase29_forward_graph import LogitForward
if __name__=="__main__":
 rows=[];prefix=[]
 for weight in [0.,.01]:
  model=load_model(weight);forward=LogitForward(model,torch.zeros(1,2,2,15,200,device="cuda"))
  for sid in [5,15,25,35]:
   x=np.load(ROOT/"phase17/cache"/f"s{sid:02d}_x.npy",mmap_mode="r");cache=np.load(ROOT/"phase28"/f"young_joint_{weight:g}"/f"s{sid:02d}.npz");raw,_=subject(sid);ii=np.linspace(0,len(x)-1,8,dtype=int);direct=preprocess(np.array(raw[ii])).reshape(-1,2,2,15,200).numpy();assert np.array_equal(direct,x[ii]);pred=[]
   for t in range(len(x)):
    with torch.no_grad():pred.append(forward.run(torch.tensor(np.array(x[t:t+1]),device="cuda")).softmax(1).cpu().numpy()[0].copy())
   pred=np.array(pred);err=float(abs(pred-cache["p"]).max());diff=int((pred.argmax(1)!=cache["p"].argmax(1)).sum());assert err<5e-5 and diff==0;rows.append(dict(weight=weight,subject=sid,windows=len(x),max_probability_error=err,argmax_disagreements=diff,preprocessing_cache_matches_raw_exact=True));print("JOINT FROZEN AUDIT",rows[-1],flush=True)
  del forward,model
  x=np.load(ROOT/"phase17/cache/s05_x.npy",mmap_mode="r")[1280:1376].copy();tokens=np.load(ROOT/"phase28/tokens/s05.npz")["tokens"][1280:1376].astype(np.int64);changed=x.copy();changed[32:]*=-1.7;changed_tokens=tokens.copy();changed_tokens[32:]=(changed_tokens[32:]+1000)%8192
  for objective in ["entropy","masked_token"]:
   m=load_model(weight);example=torch.tensor(x[:1],device="cuda");g=MaskedGraphAdapter(m,.01,example) if objective=="masked_token" else GraphAdapter(m,"entropy",.01,example);runs=[]
   for xx,tt in [(x,tokens),(changed,changed_tokens)]:
    g.reset();pred=[]
    for t in range(len(xx)):
     argument=torch.tensor(tt[t:t+1],device="cuda") if objective=="masked_token" else torch.ones(5,device="cuda")/5
     pred.append(g.step(torch.tensor(xx[t:t+1],device="cuda"),argument).detach().cpu().numpy()[0].copy())
    runs.append(np.array(pred))
   err=float(abs(runs[0][:32]-runs[1][:32]).max());assert err==0.;prefix.append(dict(weight=weight,objective=objective,prefix_windows=32,total_windows=96,max_prefix_error=err));print("PREFIX AUDIT",prefix[-1],flush=True)
   del g,m
 (OUT30/"frozen_prefix_audit.json").write_text(json.dumps(dict(status="PASS",frozen_streams=rows,prefix_cases=prefix),indent=2))

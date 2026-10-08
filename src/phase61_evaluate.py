import argparse,inspect,hashlib
import phase52_evaluate as ev
import phase55_small_lr as online
from phase52_evaluate import ROOT,np,torch,pd,json
def execute(seed):
 out=ROOT/"phase61"/f"seed{seed}";ev.OUT=out;online.OUT55=out;online.load=ev.load
 protocol=json.loads((out/"protocol.json").read_text())
 protocol["model_sha256"]={v:hashlib.sha256((out/v/"model.pt").read_bytes()).hexdigest() for v in ["current","future"]}
 (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
 rows=[]
 for v in ["current","future"]:
  m=ev.load(v)
  for sid in [5,15,25,35]:
   z=np.load(ROOT/"phase46/cache"/f"s{sid:02d}.npz")
   with torch.no_grad():
    xx=(torch.tensor(z["masked"],device="cuda")-m.recon_input_mean)/m.recon_input_scale
    yy=(torch.tensor(z["target"],device="cuda")-m.recon_target_mean)/m.recon_target_scale
    pred=m.recon_head(xx);sh=m.recon_head(torch.roll(xx,len(xx)//2,0))
    mse=float((pred-yy).square().mean());shuffle=float((sh-yy).square().mean());mean=float(yy.square().mean())
   rows.append(dict(seed=seed,variant=v,subject=sid,mse=mse,source_mean_mse=mean,shuffled_mse=shuffle,validity_pass=mse<mean and mse<shuffle))
  del m
  ev.run(v)
 pd.DataFrame(rows).to_csv(out/"reconstruction_validity.csv",index=False)
 # Same audited graph-vs-eager reference, now explicitly exercising the new heads.
 script=inspect.getsource(online.audit).replace('["initial","future","current","reconstruction"]','["future","current"]').replace('[0.,1e-5,1e-4,.001]','[0.,1e-4,.01]')
 (out/"graph_reference_source.py").write_text(script)
 path=out/"graph_audit.json"
 if path.is_symlink():path.unlink()
 exec(compile(script,str(out/"graph_reference_source.py"),"exec"),online.__dict__)
 online.audit()
 run_source=inspect.getsource(online.run).replace('(ROOT/"phase52"/variant/"model.pt")','(OUT55/variant/"model.pt")')
 (out/"online_reference_source.py").write_text(run_source)
 exec(compile(run_source,str(out/"online_reference_source.py"),"exec"),online.__dict__)
 for v,lr in protocol["online_lr"].items():online.run(v,lr)
 (out/"evaluation_complete.json").write_text(json.dumps(dict(complete=True,seed=seed,source_only=True,online_steps=protocol["online_lr"]),indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);a=p.parse_args();execute(a.seed)

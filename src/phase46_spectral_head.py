from phase30_masked_graph import *
from phase37_screen import load as load_carrier
import pandas as pd,hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
OUT=ROOT/"phase46"
class SpectralCarrier(JointCarrier):
 def __init__(self,mode):
  super().__init__("mixed");self.load_state_dict(torch.load(ROOT/"phase28/mixed_joint_0/model.pt",map_location="cpu",weights_only=False)["state"]);data=np.load(OUT/f"{mode}_readout.npz");self.reconstruction_mode=mode;self.recon_head=nn.Linear(400,24)
  with torch.no_grad():self.recon_head.weight.copy_(torch.tensor(data["coef"]));self.recon_head.bias.copy_(torch.tensor(data["intercept"]))
  for key in ["input_mean","input_scale","target_mean","target_scale"]:self.register_buffer("recon_"+key,torch.tensor(data[key],dtype=torch.float32))
 def reconstruction_loss(self,x,target):
  raw=x.flatten(0,1)
  if self.reconstruction_mode=="masked":
   mask=(torch.arange(30,device=raw.device)[None]%2==0).expand(len(raw),-1);z=masked_tokens(self.encoder,raw,mask,self.mask_token).mean(1).reshape(-1,400)
  else:z=self.encoder.forward_features(raw,input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(-1,400)
  pred=self.recon_head((z-self.recon_input_mean)/self.recon_input_scale);truth=(target-self.recon_target_mean)/self.recon_target_scale;return nn.functional.mse_loss(pred,truth)
def run():
 protocol=json.loads((OUT/"protocol.json").read_text());source=protocol["training_subjects"];val=protocol["validation_subjects"];model,_=load_carrier("mixed","unfitted");model.requires_grad_(False);state={k:v.clone() for k,v in model.state_dict().items()};sha=hashlib.sha256((ROOT/"phase28/mixed_joint_0/model.pt").read_bytes()).hexdigest();cache_audits=[]
 for sid in source+val:
  dest=OUT/"cache"/f"s{sid:02d}.npz"
  if dest.exists():
   z=np.load(dest);assert str(z["checkpoint_sha256"])==sha;continue
  root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");features=np.load(root/f"s{sid:02d}_meta.npz")["features"][:,24:48];parts={"masked":[],"unmasked":[],"classification":[]}
  with torch.inference_mode():
   for start in range(0,len(raw),128):
    x=preprocess(np.array(raw[start:start+128])).reshape(-1,2,2,15,200).cuda();flat=x.flatten(0,1);zu=model.encoder.forward_features(flat,input_chans=[0,127,128],return_patch_tokens=True).mean(1).reshape(-1,400);mask=(torch.arange(30,device="cuda")[None]%2==0).expand(len(flat),-1);zm=masked_tokens(model.encoder,flat,mask,model.mask_token).mean(1).reshape(-1,400);pr=model.head((zu-model.mean)/model.scale).softmax(1);parts["masked"].append(zm.cpu().numpy());parts["unmasked"].append(zu.cpu().numpy());parts["classification"].append(pr.cpu().numpy())
  parts={k:np.concatenate(v) for k,v in parts.items()};assert all(np.isfinite(v).all() for v in parts.values());assert np.isfinite(features).all();np.savez_compressed(dest,**parts,target=features,checkpoint_sha256=sha);print("SPECTRAL CACHE",sid,len(raw),flush=True)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());del model,state
 for sid in val:
  a=np.load(OUT/"cache"/f"s{sid:02d}.npz")["classification"];b=np.load(ROOT/"phase28/mixed_joint_0"/f"s{sid:02d}.npz")["p"];err=float(abs(a-b).max());dis=int((a.argmax(1)!=b.argmax(1)).sum());assert err<2e-5 and dis==0;cache_audits.append(dict(subject=sid,classifier_probability_error=err,argmax_disagreements=dis))
 train=[np.load(OUT/"cache"/f"s{sid:02d}.npz") for sid in source];target=np.concatenate([z["target"] for z in train]);sy=StandardScaler().fit(target);ym=sy.mean_.astype(np.float32);ys=sy.scale_.astype(np.float32);yt=(target-ym)/ys;sample_weights=np.concatenate([np.full(len(z["target"]),1/len(z["target"])) for z in train]);sample_weights/=sample_weights.mean();rows=[];readouts=[]
 for mode in ["masked","unmasked"]:
  z=np.concatenate([v[mode] for v in train]);sx=StandardScaler().fit(z);xm=sx.mean_.astype(np.float32);xs=sx.scale_.astype(np.float32);xnorm=(z-xm)/xs;head=Ridge(alpha=100).fit(xnorm,yt,sample_weight=sample_weights);np.savez(OUT/f"{mode}_readout.npz",coef=head.coef_.astype(np.float32),intercept=head.intercept_.astype(np.float32),input_mean=xm,input_scale=xs,target_mean=ym,target_scale=ys,parent_sha256=sha);readouts.append(dict(mode=mode,parameters=24*401,training_windows=len(z),alpha=100,additional_class_labels=0));tt=torch.tensor(head.coef_,dtype=torch.float32);bb=torch.tensor(head.intercept_,dtype=torch.float32)
  for sid in val:
   q=np.load(OUT/"cache"/f"s{sid:02d}.npz");xx=(q[mode]-xm)/xs;yy=(q["target"]-ym)/ys;pred=head.predict(xx);reference=nn.functional.linear(torch.tensor(xx),tt,bb).numpy();error=float(abs(pred-reference).max());assert error<5e-4 and np.linalg.norm(pred-reference)/max(np.linalg.norm(pred),1e-12)<1e-5;shuffled=head.predict(np.roll(xx,len(xx)//2,axis=0));mse=float(np.square(pred-yy).mean());mean_mse=float(np.square(yy).mean());shuffle_mse=float(np.square(shuffled-yy).mean());rows.append(dict(mode=mode,subject=sid,windows=len(xx),mse=mse,source_mean_mse=mean_mse,shuffled_mse=shuffle_mse,relative_mse_vs_mean=mse/mean_mse,readout_cpu_torch_error=error,validity_pass=mse<mean_mse and mse<shuffle_mse))
 df=pd.DataFrame(rows);df.to_csv(OUT/"validation.csv",index=False);decision=dict(primary="masked",primary_usable=bool(df[df["mode"]=="masked"].validity_pass.all()),unmasked_usable=bool(df[df["mode"]=="unmasked"].validity_pass.all()),criterion=protocol["validity"]);(OUT/"validity_decision.json").write_text(json.dumps(decision,indent=2));(OUT/"audit.json").write_text(json.dumps(dict(status="PASS",classification_labels_used=False,classifier_state_unchanged_exact=True,parent_checkpoint_sha256=sha,readouts=readouts,classification_prediction_audits=cache_audits,readout_predictions_torch_checked=True),indent=2));style='<style>body{font:15px/1.65 system-ui;max-width:1200px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style>'
 html='<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>当前窗口频谱重建：额外小型辅助读出</h1><p>固定混合来源分类器及原掩码向量，拟合400→24线性读出，预测当前30秒窗口两导联各12个频带的log相对功率。输入和目标缩放仅用10名来源训练受试者，Ridge固定alpha100、按人等权；不使用新增分类标签、不选择目标人群参数。比较固定50%掩码池化特征和未掩码池化特征。</p><p>这是新的辅助目标和额外9624个参数拟合，不能称原有重建头零成本复用。掩码放在卷积patch嵌入之后，与既有分支一致；前端GroupNorm统计仍可涉及整个当前窗口，因此不是严格遮蔽原始波形的任务。输入打乱和来源均值用于检验输入相关性，不代表在线分类收益。</p>'+df.to_html(index=False,float_format=lambda v:f"{v:.6f}")+f'<p>预定掩码辅助有效性条件是否满足：{decision["primary_usable"]}。分类器参数保持完全相同，四条源验证分类预测核对通过。后续必须单独验证新鲜更新与正确数，不能由重建MSE推断TTT有效。</p><a href="audit.json">审计</a> · <a href="validity_decision.json">有效性判定</a> · <a href="protocol.json">协议</a></html>';(OUT/"report.html").write_text(html);print(df.to_string(index=False));print(decision)
if __name__=="__main__":run()

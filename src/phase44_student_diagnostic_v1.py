from phase30_masked_graph import *
from phase37_screen import load as load_carrier
import pandas as pd
OUT=ROOT/"phase44"
def manual(model,x,mode):
 e=model.encoder;raw=x.flatten(0,1)
 if mode=="duplicate16":raw=raw.repeat(1,8,1,1)
 batch,n,t,_=raw.shape;mask=(torch.arange(n*t,device=raw.device)[None]%2==0).expand(batch,-1);z=e.patch_embed(raw);z=torch.where(mask[:,:,None],model.mask_token.expand(batch,n*t,-1),z);z=torch.cat([e.cls_token.expand(batch,-1,-1),z],1);pos=e.pos_embed[:,[0]+[127,128]*(n//2)];spatial=pos[:,1:].unsqueeze(2).expand(batch,-1,t,-1).flatten(1,2);spatial=torch.cat([pos[:,:1].expand(batch,-1,-1),spatial],1);z=z+spatial;z[:,1:]+=e.time_embed[:,:t].unsqueeze(1).expand(batch,n,-1,-1).flatten(1,2);z=e.pos_drop(z);bias=None;attention=[]
 if mode=="cls8":bias=torch.zeros((z.shape[1],z.shape[1]),device=z.device);bias[:,0]=-np.log(8)
 for i,blk in enumerate(e.blocks):
  if i in [0,11]:
   a=blk(z,rel_pos_bias=bias,return_attention=True)[:,:,1:,0];attention.append(dict(layer=i+1,patch_to_cls=float(a.mean()),masked_to_cls=float(a[:,:,mask[0]].mean())))
  z=blk(z,rel_pos_bias=bias)
 z=e.norm(z)[:,1:31];m=(torch.arange(30,device=z.device)[None]%2==0).expand(batch,-1);return model.aux_head(z[m]),attention
def run():
 rows=[];attrows=[];audit=[];protocol=json.loads((OUT/"protocol.json").read_text())
 for carrier in protocol["carriers"]:
  source=carrier.split("_")[0] if carrier!="pretrained" else "young"
  model=JointCarrier(source).cuda().eval() if carrier=="pretrained" else load_carrier(source,carrier.split("_")[1])[0];model.requires_grad_(False);state={k:v.clone() for k,v in model.state_dict().items()}
  for sid in protocol["subjects"]:
   root=ROOT/("data/processed" if sid<20 else "phase2/processed");raw=np.load(root/f"s{sid:02d}_x.npy",mmap_mode="r");idx=np.linspace(0,len(raw)-1,32,dtype=int);x=preprocess(np.array(raw[idx])).reshape(-1,2,2,15,200);tokens=torch.tensor(np.load(ROOT/"phase28/tokens"/f"s{sid:02d}.npz")["tokens"][idx].astype(np.int64));old=torch.tensor(np.load(ROOT/"phase26/tokens"/f"s{sid:02d}.npz")["tokens"][idx].astype(np.int64));mask=(torch.arange(30)[None]%2==0).expand(64,-1);truth=tokens.flatten(0,1)[mask];oldtruth=old.flatten(0,1)[mask];outputs={}
   with torch.inference_mode():
    before=model(x[:1].cuda()).cpu().clone()
    for mode in ["original","cls8","duplicate16"]:
     logits=[];shuffled=[];local=[]
     for i in range(0,len(x),4):
      z,a=manual(model,x[i:i+4].cuda(),mode);logits.append(z.cpu());local.extend(a)
      if mode!="duplicate16":
       zz,_=manual(model,x[(np.arange(i,i+4)+16)%32].cuda(),mode);shuffled.append(zz.cpu())
     logits=torch.cat(logits);outputs[mode]=logits;loss=float(nn.functional.cross_entropy(logits,truth));accuracy=float((logits.argmax(1)==truth).float().mean());oldloss=float(nn.functional.cross_entropy(logits,oldtruth));shuffleloss=float(nn.functional.cross_entropy(torch.cat(shuffled),truth)) if shuffled else rows[-1]["shuffled_ce"];pred=logits.argmax(1);dominant=float(torch.bincount(pred,minlength=8192).max()/len(pred));rows.append(dict(carrier=carrier,subject=sid,student_attention=mode,corrected_teacher_ce=loss,corrected_teacher_accuracy=accuracy,original_teacher_ce=oldloss,shuffled_ce=shuffleloss,prediction_dominant_fraction=dominant))
     for layer in [1,12]:
      aa=[v for v in local if v["layer"]==layer];attrows.append(dict(carrier=carrier,subject=sid,student_attention=mode,layer=layer,patch_to_cls=float(np.mean([v["patch_to_cls"] for v in aa])),masked_to_cls=float(np.mean([v["masked_to_cls"] for v in aa]))))
     if mode=="original":
      ref,_=model.auxiliary(x[:4].cuda(),tokens[:4].cuda(),False);assert abs(float(ref)-float(nn.functional.cross_entropy(logits[:120],truth[:120])))<2e-5
    after=model(x[:1].cuda()).cpu();assert torch.equal(before,after)
   err=float(abs(outputs["cls8"]-outputs["duplicate16"]).max());agreement=float((outputs["cls8"].argmax(1)==outputs["duplicate16"].argmax(1)).float().mean());assert err<2e-4 and agreement>=.999;audit.append(dict(carrier=carrier,subject=sid,cls8_duplicate_logit_error=err,cls8_duplicate_argmax_agreement=agreement,classifier_prediction_unchanged_exact=True));print("STUDENT CLS",carrier,sid,[round(v["corrected_teacher_ce"],4) for v in rows[-3:]],err,flush=True)
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items());del model,state
 pd.DataFrame(rows).to_csv(OUT/"participants.csv",index=False);pd.DataFrame(attrows).to_csv(OUT/"attention.csv",index=False);summary=pd.DataFrame(rows).groupby(["carrier","student_attention"]).mean(numeric_only=True).drop(columns="subject").reset_index();summary.to_csv(OUT/"summary.csv",index=False);(OUT/"audit.json").write_text(json.dumps(dict(status="PASS",training=False,classification_labels_used=False,weights_unchanged=True,manual_auxiliary_forward_verified=True,cases=audit),indent=2));style='<style>body{font:15px/1.65 system-ui;max-width:1250px;margin:30px auto}td,th{padding:5px;border-bottom:1px solid #ddd}</style>';(OUT/"report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>学生掩码分支的CLS注意力兼容性：只读诊断</h1><p>教师已采用CLS键注意力减去log8；学生辅助分支此前保持原注意力。本次比较原学生、同样CLS8修正、重复16个同位置通道的数学等价对照。五个冻结载体、四名源验证受试者各32窗口，不使用分类标签、不训练、不修改分类器或实际在线分支。</p><p>重复通道只是注意力等价性检验，不是新的生理导联。记录目标CE、输入打乱对照、原退化教师目标CE及第1/12层CLS注意力。辅助CE即使改善，也不能推断在线分类收益。</p>'+summary.to_html(index=False,float_format=lambda v:f"{v:.5f}")+'<p><a href="attention.csv">注意力统计</a> · <a href="audit.json">状态与等价性审计</a> · <a href="protocol.json">协议</a></p></html>');print(summary.to_string(index=False))
if __name__=="__main__":run()

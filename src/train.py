from common import *
import argparse,time,copy
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['spectral','eegnet'],default='spectral');p.add_argument('--fold',default='A');p.add_argument('--seed',type=int,default=17);p.add_argument('--epochs',type=int,default=25);args=p.parse_args()
seed_all(args.seed);split=CFG['folds'][args.fold];out=ROOT/f'results/{args.kind}_{args.fold}_{args.seed}';out.mkdir(exist_ok=True)
import fcntl
lock=(out/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
if (out/'training_complete.json').exists():
    print('TRAIN COMPLETE already saved',flush=True);raise SystemExit(0)
xs=[];ys=[]
for sid in split['base_train']:
    x,y=load_subject(sid,args.kind);xs.append(np.array(x[y>=0]));ys.append(y[y>=0])
x=torch.from_numpy(np.concatenate(xs));y=torch.from_numpy(np.concatenate(ys));del xs,ys
model=make_model(args.kind)
if args.kind=='spectral':model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp(min=.1))
counts=torch.bincount(y,minlength=5).float();weights=(counts.sum()/counts.clamp(min=1)).sqrt();weights/=weights.mean()
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
val={s:load_subject(s,args.kind) for s in split['base_val']}
best=-1;history=[];start=time.perf_counter();batch=128 if args.kind=='spectral' else 64
print('START',vars(args),'examples',len(y),'params',sum(p.numel() for p in model.parameters()),'affinity',sorted(os.sched_getaffinity(0)),flush=True)
start_epoch=0
if (out/'resume.pt').exists():
    resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);history=resume['history'];best=resume['best'];start_epoch=resume['epoch']
    print('RESUME',start_epoch,flush=True)
for epoch in range(start_epoch,args.epochs):
    t=time.perf_counter();model.train();perm=torch.randperm(len(y));total=0
    for idx in perm.split(batch):
        if len(idx)<2:continue
        opt.zero_grad();logits=model(x[idx]);loss=nn.functional.cross_entropy(logits,y[idx],weight=weights);loss.backward();opt.step()
        if args.kind=='eegnet':
            with torch.no_grad():
                w=model.body[2].weight;w.mul_((1./w.flatten(1).norm(dim=1).clamp(min=1.)).view(-1,1,1,1))
        total+=float(loss.detach())*len(idx)
    scores={sid:metrics(vy,predict(model,vx)) for sid,(vx,vy) in val.items()}
    score=float(np.mean([a['macro_f1'] for a in scores.values()]))
    row={'epoch':epoch+1,'loss':total/len(y),'val_macro_f1':score,'seconds':time.perf_counter()-t,'val':scores};history.append(row)
    print(json.dumps(row),flush=True)
    if score>best:
        best=score;torch.save({'state':copy.deepcopy(model.state_dict()),'kind':args.kind,'fold':args.fold,'seed':args.seed,'epoch':epoch+1,'val_f1':score,'config':CFG},out/'model.pt')
    (out/'train.json').write_text(json.dumps({'history':history,'elapsed':time.perf_counter()-start,'train_subjects':split['base_train'],'val_subjects':split['base_val'],'affinity':sorted(os.sched_getaffinity(0))},indent=2))
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'history':history,'best':best,'epoch':epoch+1},out/'resume.tmp')
    (out/'resume.tmp').replace(out/'resume.pt')
(out/'training_complete.json').write_text(json.dumps({'best_val_f1':best,'epochs':args.epochs,'seed':args.seed,'fold':args.fold,'kind':args.kind}))
print('TRAIN COMPLETE',best,'seconds',time.perf_counter()-start,flush=True)

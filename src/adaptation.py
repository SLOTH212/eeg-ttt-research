from common import *
import copy,time

def entropy(p):return -(p*p.clamp(min=1e-9).log()).sum(-1)
class Adapter:
    def __init__(self,model,lr=.001,radius=.1):
        self.model=copy.deepcopy(model).eval();self.model.requires_grad_(False)
        self.params=[];self.names=[];self.head_only=getattr(model,'writable_scope','affine')=='head'
        if getattr(model,'writable_scope','affine')=='head':
            for n,p in self.model.head.named_parameters():p.requires_grad_(True);self.params.append(p);self.names.append('head.'+n)
        else:
            for name,mod in self.model.named_modules():
                if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):
                    for n,p in mod.named_parameters():p.requires_grad_(True);self.params.append(p);self.names.append(name+'.'+n)
        self.source=self.snapshot();self.lr=lr;self.radius=radius
        # Per-tensor RMS scale floor 1 prevents near-zero bias from prohibiting any update.
        self.scales=[max(float(p.square().mean().sqrt()),1.) for p in self.source]
    def snapshot(self):return [p.detach().clone() for p in self.params]
    def restore(self,s):
        with torch.no_grad():
            for p,v in zip(self.params,s):p.copy_(v)
    def drift(self):return float(sum(((p-s)/scale).square().sum() for p,s,scale in zip(self.params,self.source,self.scales)).sqrt().detach())
    def candidate(self,x):
        self.model.zero_grad(set_to_none=True)
        if self.head_only:
            with torch.no_grad():self.current_features=self.model.features(x)
            logits=self.model.head(self.current_features)
        else:logits=self.model(x)
        prob=logits.softmax(1);loss=entropy(prob).mean();loss.backward()
        old=self.snapshot();gradnorm=float(sum(p.grad.square().sum() for p in self.params).sqrt())
        drift=self.drift()
        with torch.no_grad():
            for p,s,scale in zip(self.params,self.source,self.scales):
                p.add_(p.grad,alpha=-self.lr)
                d=p-s;limit=self.radius*scale*np.sqrt(p.numel());norm=float(d.norm())
                if norm>limit:p.copy_(s+d*(limit/norm))
            candidate=self.snapshot();after=(self.model.head(self.current_features) if self.head_only else self.model(x)).softmax(1)
        signals=np.array([float(loss.detach()),gradnorm,float((after-prob.detach()).abs().sum()),drift,float(prob.max().detach()),float(entropy(after).mean()),*prob.detach().numpy()[0]],dtype='float64')
        self.restore(old)
        return prob.detach().numpy()[0],candidate,signals

def stream(model,x,mode,lr=.001,gate=None,history_features=None,collect=False,subject=-1,stride=20):
    """No y argument: all decisions depend only on current/past inputs. Diagnostics stored separately."""
    a=Adapter(model,lr);pred=[];times=[];actions=[];diagnostics=[];previous=None
    for t in range(len(x)):
        start=time.perf_counter();xt=torch.tensor(np.array(x[t:t+1]),dtype=torch.float32)
        prob,candidate,s=a.candidate(xt)
        change=0. if previous is None else float(np.linalg.norm(history_features[t]-previous)/np.sqrt(len(previous)))
        features=np.r_[s,change,float(previous is None)]
        if collect and t%stride==0 and t+10<len(x):
            old=a.snapshot();future=torch.tensor(np.array(x[t+1:t+11]),dtype=torch.float32)
            with torch.no_grad():
                forward=a.model.head if a.head_only else a.model
                fx=a.model.features(future) if a.head_only else future
                cx=a.current_features if a.head_only else xt
                keep=forward(fx).softmax(1).numpy();current_keep=forward(cx).softmax(1).numpy()[0]
                a.restore(candidate);write=forward(fx).softmax(1).numpy();current_write=forward(cx).softmax(1).numpy()[0]
                a.restore(a.source);reset=forward(fx).softmax(1).numpy();current_reset=forward(cx).softmax(1).numpy()[0]
                a.restore(old)
            diagnostics.append({'t':t,'features':features,'keep':keep,'write':write,'reset':reset,'current_keep':current_keep,'current_write':current_write,'current_reset':current_reset})
        action=0
        if mode=='continuous':action=1
        elif mode=='periodic':action=int(t%20==0)
        elif mode=='low_entropy':action=int(s[0]<.5)
        elif mode=='reset_periodic':action=2 if (t+1)%100==0 else 1
        elif mode=='controller':action=int(gate(features))
        elif mode=='random':action=int((t*1103515245+12345)%7<2)
        elif mode=='frozen':action=0
        else:raise ValueError(mode)
        pred.append(prob) # emitted before action
        if action==1:a.restore(candidate)
        elif action==2:a.restore(a.source)
        actions.append(action)
        if history_features is not None:previous=history_features[t].copy()
        times.append(time.perf_counter()-start)
    return np.array(pred),diagnostics,{'actions':np.bincount(actions,minlength=3).tolist(),'seconds':sum(times),'latency_ms_p50':1000*float(np.quantile(times,.5)),'latency_ms_p95':1000*float(np.quantile(times,.95)),'diagnostic_cost_included':collect,'writable_params':sum(p.numel() for p in a.params)}

def score_diagnostics(ds,y,subject,trajectory):
    rows=[]
    for d in ds:
        t=d['t']
        for h in [1,4,10]:
            yy=y[t+1:t+1+h];mask=yy>=0
            if not mask.any():continue
            losses={k:float(-np.log(np.clip(d[k][:h][mask,np.array(yy[mask])],1e-9,1.)).mean()) for k in ['keep','write','reset']}
            rows.append({'subject':subject,'t':t,'h':h,'trajectory':trajectory,'features':d['features'].tolist(),'keep_loss':losses['keep'],'write_gain':losses['keep']-losses['write'],'reset_gain':losses['keep']-losses['reset'],'current_write_gain':float(np.log(np.clip(d['current_write'][y[t]],1e-9,1))-np.log(np.clip(d['current_keep'][y[t]],1e-9,1))) if y[t]>=0 else None,'current_reset_gain':float(np.log(np.clip(d['current_reset'][y[t]],1e-9,1))-np.log(np.clip(d['current_keep'][y[t]],1e-9,1))) if y[t]>=0 else None,'transition':bool(y[t]>=0 and np.any(yy[mask]!=y[t]))})
    return rows

def smooth(p,alpha=.8):
    z=p[0].copy();out=[]
    for row in p:z=alpha*row+(1-alpha)*z;out.append(z.copy())
    return np.array(out)

def t3a(model,x,k=20):
    """Official T3A support rule, moved AFTER prediction to meet causal protocol."""
    model.eval()
    with torch.no_grad():
        zs=torch.cat([model.features(torch.tensor(np.array(x[a:a+256]),dtype=torch.float32)) for a in range(0,len(x),256)])
        supports=model.head.weight.detach().clone();warm=model.head(supports).softmax(1);labels=warm.argmax(1);ent=entropy(warm)
        probs=[]
        for z in zs:
            weights=nn.functional.normalize(supports,dim=1).T@nn.functional.one_hot(labels,5).float()
            weights=nn.functional.normalize(weights,dim=0)
            probs.append((z@weights).softmax(0).numpy())
            pp=model.head(z).softmax(0);supports=torch.cat([supports,z[None]]);labels=torch.cat([labels,pp.argmax().view(1)]);ent=torch.cat([ent,entropy(pp).view(1)])
            ii=torch.cat([torch.where(labels==c)[0][torch.argsort(ent[labels==c])[:k]] for c in range(5)])
            supports=supports[ii];labels=labels[ii];ent=ent[ii]
    return np.array(probs)

def causal_bn(model,x):
    """Prediction with prior running stats, then EMA stats refresh using current window only; no gradients."""
    m=copy.deepcopy(model).eval();out=[]
    hooks=[]
    def hook(mod,args):
        v=args[0];dims=(0,)+tuple(range(2,v.ndim));mean=v.mean(dims);var=v.var(dims,unbiased=False)
        # Defer updates until complete prediction to avoid mutation during forward.
        pending.append((mod,mean,var))
    for mod in m.modules():
        if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d)):hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for row in x:
            pending=[];out.append(m(torch.tensor(np.array(row[None]),dtype=torch.float32)).softmax(1).numpy()[0])
            for mod,mean,var in pending:
                delta=mean-mod.running_mean
                mod.running_var.mul_(.99).add_(var,alpha=.01).add_(delta.square(),alpha=.0099)
                mod.running_mean.lerp_(mean,.01)
    for h in hooks:h.remove()
    return np.array(out)

import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,set(range(8,24)))
from pathlib import Path
import json, random, numpy as np, torch
from torch import nn
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, log_loss
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/pilot.json').read_text())
torch.set_num_threads(int(os.environ.get('EEG_THREADS','4')))
torch.set_num_interop_threads(1)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def metrics(y,p):
    k=y>=0;y=y[k];p=p[k];pred=p.argmax(1)
    cm=confusion_matrix(y,pred,labels=range(5))
    return {'n':len(y),'macro_f1':float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)), 'kappa':(None if len(np.unique(y))==1 and np.array_equal(y,pred) else float(cohen_kappa_score(y,pred))), 'ce':float(log_loss(y,p,labels=range(5))), 'recall':(cm.diagonal()/np.maximum(cm.sum(1),1)).tolist(),'confusion':cm.tolist()}
def load_subject(sid,kind):
    d=np.load(ROOT/f'data/processed/s{sid:02d}_meta.npz')
    x=d['features'] if kind=='spectral' else np.load(ROOT/f'data/processed/s{sid:02d}_x.npy',mmap_mode='r')
    return x,d['y']
class SpectralNet(nn.Module):
    """Additional cheap probe, NOT EEGNet: fixed within-window PSD + BN MLP."""
    def __init__(self,dim=52):
        super().__init__();self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.body=nn.Sequential(nn.Linear(dim,64),nn.BatchNorm1d(64),nn.ELU(),nn.Dropout(.25),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ELU())
        self.head=nn.Linear(32,5)
    def features(self,x):return self.body((x-self.mean)/self.scale)
    def forward(self,x):return self.head(self.features(x))
class EEGNet(nn.Module):
    """EEGNet-style 8,2; 100Hz kernel=51; adaptive average pool=8 instead of flatten; PyTorch BN eps=.001,momentum=.01. Not a reproduction claim."""
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(1,8,(1,51),padding=(0,25),bias=False),nn.BatchNorm2d(8,eps=.001,momentum=.01),nn.Conv2d(8,16,(2,1),groups=8,bias=False),nn.BatchNorm2d(16,eps=.001,momentum=.01),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.25),nn.Conv2d(16,16,(1,15),padding=(0,7),groups=16,bias=False),nn.Conv2d(16,16,1,bias=False),nn.BatchNorm2d(16,eps=.001,momentum=.01),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(.25),nn.AdaptiveAvgPool2d((1,8)),nn.Flatten())
        self.head=nn.Linear(128,5)
    def features(self,x):return self.body(x.unsqueeze(1))
    def forward(self,x):return self.head(self.features(x))
def make_model(kind):return SpectralNet() if kind=='spectral' else EEGNet()
def predict(m,x,batch=256):
    m.eval();out=[]
    with torch.no_grad():
        for a in range(0,len(x),batch):out.append(m(torch.tensor(np.array(x[a:a+batch]),dtype=torch.float32)).softmax(1).numpy())
    return np.concatenate(out)
def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=make_model(ck['kind']);m.load_state_dict(ck['state']);m.writable_scope=ck.get('writable_scope','affine');m.eval();return m,ck

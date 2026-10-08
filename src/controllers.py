import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
class Gate:
    def __init__(self,reg,indices):self.reg=reg;self.indices=indices
    def values(self,f):return self.reg.predict(np.atleast_2d(f)[:,self.indices])
    def __call__(self,f):return np.r_[0.,self.values(f)[0]].argmax()
def fit_gate(rows,typ='future'):
    rows=[r for r in rows if r['h']==4 and (typ!='current' or r['current_write_gain'] is not None)]
    xx=np.array([r['features'] for r in rows]);prefix='current_' if typ=='current' else ''
    yy=np.array([[r[prefix+'write_gain'],r[prefix+'reset_gain']] for r in rows]);indices=[0,4] if typ=='confidence' else list(range(xx.shape[1]))
    reg=make_pipeline(StandardScaler(),Ridge(alpha=100.));ss=np.array([r['subject'] for r in rows]);weights=np.array([len(ss)/(len(set(ss))*sum(ss==s)) for s in ss])
    reg.fit(xx[:,indices],yy,ridge__sample_weight=weights);return Gate(reg,indices)

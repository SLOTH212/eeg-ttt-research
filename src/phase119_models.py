from common import *
from phase77_predict import context
class ResidualLogitHead(SpectralNet):
 def __init__(self,dim):
  super().__init__(dim)
  nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):return x[:,-5:]+super().forward(x)
def transform(logits,variant):
 if variant=="current":return logits
 if variant=="repeat11":return np.tile(logits,(1,11))
 if variant=="history11":return context(logits)
 raise ValueError(variant)

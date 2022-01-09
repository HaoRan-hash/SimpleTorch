import numpy as np
import math
from .operators import LinearFunction, Operator
from SimpleTorch import Tensor


class Module:
    def __init__(self):
        self.parameters = []
    
    def __setattr__(self, key, value):
        if isinstance(value, Module):
            for name, param in value.parameters:
                self.parameters.append((key+'_'+name, param))
        self.__dict__[key] = value
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name, param):
        self.parameters.append((name, param))
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        k = math.sqrt(1 / in_features)
        self.w = Tensor(np.random.uniform(-k, k, size=(out_features, in_features)), requires_grad=True)
        self.b = Tensor(np.random.uniform(-k, k, size=(out_features, )), requires_grad=True)

        self.register_parameter('weight', self.w)
        self.register_parameter('bias', self.b)

    def forward(self, x):
        linear_func = LinearFunction()
        y = linear_func(x, self.w, self.b)

        return y


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()

        for i, arg in enumerate(args):
            if not isinstance(arg, (Module, Operator)):
                raise TypeError('args must be Module or Operator')

            if isinstance(arg, Module):
                for name, param in arg.parameters:
                    self.register_parameter(str(i)+'_'+name, param)
        
        self.args = args   # type(args): tuple
    
    def forward(self, x):
        for arg in self.args:
            x = arg(x)
        y = x

        return y

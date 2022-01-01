class SGD:
    def __init__(self, parameters, lr):
        """
        type(parameters): List[Tuple(str, Tensor)]
        type(lr): float
        """
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for _, param in self.parameters:
            param.grad[:] = 0.
    
    def step(self):
        for _, param in self.parameters:
            param.data = param.data - self.lr * param.grad

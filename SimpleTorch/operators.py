import numpy as np
from SimpleTorch import Tensor, global_state


class Operator:
    def __init__(self):
        self.ctx = []
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Operator):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        type(x): Tensor
        rtype: Tensor
        """
        mask = (x.data > 0)
        res = x.data * mask

        if not global_state['grad_enabled'] or (not x.requires_grad):
            res = Tensor(res, requires_grad=False)
            res.is_leaf = False
            return res

        self.ctx.append(mask)   # 存的mask是一个ndarray
        res = Tensor(res, requires_grad=True)
        res.is_leaf = False
        res.grad_fn = self.backward
        res.pre.append(x)

        return res
    
    def backward(self, grad_output):
        """
        type(grad_output): ndarray
        rtype: ndarray
        """
        mask, = self.ctx
        grad_x = grad_output * mask

        # 释放计算图和资源
        self.ctx.clear()

        return grad_x


class LinearFunction(Operator):
    def __init__(self):
        super(LinearFunction, self).__init__()

    def forward(self, x, w, b):
        """
        type(x): Tensor
        type(w): Tensor
        type(b): Tensor
        rtype: Tensor
        """
        res = np.dot(x.data, w.data.T) + b.data

        if not global_state['grad_enabled'] or (not x.requires_grad and not w.requires_grad and not b.requires_grad):
            res = Tensor(res, requires_grad=False)
            res.is_leaf = False
            return res

        self.ctx += [x.data, w.data]   # 存的是ndarray
        res = Tensor(res, requires_grad=True)
        res.is_leaf = False
        res.grad_fn = self.backward
        res.pre += [x, w, b]

        return res

    def backward(self, grad_output):
        """
        type(grad_output): ndarray
        rtype: ndarray, ndarray, ndarray
        """
        x_data, w_data = self.ctx
        grad_x = np.matmul(grad_output, w_data)
        grad_w = np.matmul(grad_output.T, x_data)
        grad_b = np.sum(grad_output, axis=0)

        # 释放计算图和资源
        self.ctx.clear()

        return grad_x, grad_w, grad_b


class MSELoss(Operator):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y):
        """
        type(y_pred): Tensor
        type(y): Tensor
        rtype: Tensor
        """
        temp = y_pred.data - y.data

        res = np.mean(temp * temp)   # 这里的type(res)=np.float32
        res = np.array([res])

        if not global_state['grad_enabled'] or (not y_pred.requires_grad and not y.requires_grad):
            res = Tensor(res, requires_grad=False)
            res.is_leaf = False
            return res

        self.ctx.append(temp)
        res = Tensor(res, requires_grad=True)
        res.is_leaf = False
        res.grad_fn = self.backward
        res.pre += [y_pred, y]

        return res
    
    def backward(self, grad_output):
        """
        type(grad_output): float
        rtype: ndarray
        """
        temp, = self.ctx
        n = temp.shape[0] * temp.shape[1]
        grad_y_pred = grad_output * 2 * temp / n

        # 释放资源
        self.ctx.clear()

        return grad_y_pred, None


class CrossEntropyLoss(Operator):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, y_pred, y):
        """
        type(y_pred): Tensor
        type(y): Tensor
        rtype: Tensor
        """
        if y_pred.data.shape != y.data.shape:
            raise RuntimeError('y_pred.shape must be same as y.shape')
    
        y_pred_exp = np.exp(y_pred.data)
        y_pred_sum = np.sum(y_pred_exp, axis=1, keepdims=True)
        softmax_res = y_pred_exp / y_pred_sum
        log_softmax = np.log(softmax_res)
        res = np.mean(-np.sum(log_softmax * y.data, axis=1))   # 这里的type(res)=np.float32
        res = np.array([res])

        if not global_state['grad_enabled'] or (not y_pred.requires_grad and not y.requires_grad):
            res = Tensor(res, requires_grad=False)
            res.is_leaf = False
            return res

        self.ctx += [softmax_res, y.data]
        res = Tensor(res, requires_grad=True)
        res.is_leaf = False
        res.grad_fn = self.backward
        res.pre += [y_pred, y]

        return res
    
    def backward(self, grad_output):
        """
        type(grad_output): float
        rtype: ndarray
        """
        softmax_res, y_data = self.ctx
        n = softmax_res.shape[0]
        grad_y_pred = grad_output * (softmax_res - y_data) / n

        # 释放资源
        self.ctx.clear()

        return grad_y_pred, None

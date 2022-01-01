import numpy as np


global_state = {'grad_enabled': True}


class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        type(data): list or ndarray
        """
        # # 处理list
        if isinstance(data, list):
            data = np.array(data)

        if not isinstance(data, np.ndarray):
            raise TypeError('type(data) must be ndarray or list')
        if requires_grad and not np.issubdtype(data.dtype, np.floating):
            raise RuntimeError('Only Tensor of floating point can requires gradient')

        self.data = data
        self.requires_grad = requires_grad

        if not self.requires_grad:
            self.grad = None
        else:
            self.grad = np.zeros_like(self.data)
        
        # 用户不能指定这些属性
        self.is_leaf = True
        self.grad_fn = None
        self.pre = []

    def backward(self, grad_output=np.array([1.], dtype=np.float32)):
        """
        type(grad_output): float or ndarray
        rtype: None
        """
        if not self.requires_grad:
            raise RuntimeError(f'This Tensor.requires_grad=False')
        if self.is_leaf:
            raise RuntimeError(f'This Tensor is leaf tensor')
        if self.data.shape != grad_output.shape:
            raise RuntimeError(f'grad_output shape must be same as the Tensor shape')
        
        queue = [(self, grad_output)]
        while queue:
            item = queue.pop()
            cur_node, grad = item

            grads = cur_node.grad_fn(grad)
            if not isinstance(grads, tuple):
                grads = (grads, )

            for node, grad in zip(cur_node.pre, grads):
                if not isinstance(grad, np.ndarray) or not node.requires_grad:   # 如果这条grad为None 或 该节点不需要导数
                    continue

                if node.is_leaf:   # 如果是叶子节点
                    node.grad += grad
                else:  
                    queue.append((node, grad))
            # 释放计算图
            cur_node.pre.clear()


class no_grad:
    def __enter__(self):
        global_state['grad_enabled'] = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global_state['grad_enabled'] = True

from abc import abstractmethod
import cupy as cp

class Layer:
    def __init__(self) -> None:
        self.optimizable = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

class Linear(Layer):
    def __init__(self, in_features, out_features, weight_decay=False, weight_decay_lambda=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Xavier初始化
        scale = cp.sqrt(6.0 / (in_features + out_features))
        self.W = cp.random.uniform(-scale, scale, (in_features, out_features))
        self.b = cp.zeros((1, out_features))
        # # He Initialization
        # self.W = cp.random.randn(in_features, out_features).astype(cp.float32) * cp.sqrt(2 / in_features)
        # self.b = cp.ones((1, out_features), dtype=cp.float32)
        self.X = None

        self.grads = {'W' : None, 'b' : None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def parameters(self):
        return [
            ('W', self.W, self.grads['W']),
            ('b', self.b, self.grads['b'])
        ]

    def forward(self, X):
        self.X = X
        return cp.dot(X, self.W) + self.b
    
    def backward(self, grads):
        # grads: (batch_size, out_features)
        self.grads['W'] = cp.dot(self.X.T, grads) / self.X.shape[0]
        self.grads['b'] = cp.sum(grads, axis=0, keepdims=True) / self.X.shape[0]

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return cp.dot(grads, self.W.T)
    
    
class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return cp.maximum(0, X)

    def backward(self, grad):
        return cp.where(grad > 0, grad, 0)

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.X = None
    def forward(self, X):
        self.X = X
        return cp.where(X > 0, X, self.alpha * X)
    def backward(self, grad):
        return cp.where(self.X > 0, grad, self.alpha * grad)
    
class Logistic(Layer):
    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        self.X = X
        return 1 / (1 + cp.exp(-X))

    def backward(self, grad):
        return grad * self.X * (1 - self.X)

class MultiCrossEntropyLoss(Layer):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.X: cp.ndarray = None
        self.y: cp.ndarray = None

    def forward(self, X: cp.ndarray, y: cp.ndarray):
        # X: (batch_size, num_classes), y: (batch_size,)
        self.X = X
        self.y = y
        batch_size = X.shape[0]
        # 计算 softmax
        exp_X = cp.exp(X - cp.max(X, axis=1, keepdims=True))
        softmax_X = exp_X / cp.sum(exp_X, axis=1, keepdims=True)
        # 计算交叉熵损失
        loss = -cp.sum(cp.log(softmax_X[cp.arange(batch_size), y])) / batch_size
        return loss
    
    def backward(self):
        # 计算梯度
        batch_size = self.X.shape[0]
        grad = self.X.copy()
        grad[cp.arange(batch_size), self.y] -= 1
        grad /= batch_size
        self.model.backward(grad)
        
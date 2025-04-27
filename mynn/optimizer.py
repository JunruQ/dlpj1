from abc import abstractmethod
import numpy as np
import cupy as cp

class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, init_lr, model, momentum=0.0):
        """
        Stochastic Gradient Descent optimizer with optional momentum.
        
        Args:
            init_lr (float): Initial learning rate.
            model: Model with parameters() method returning list of (layer, param_name, param, grad) tuples.
            momentum (float): Momentum factor, default is 0.0 (no momentum).
        """
        super().__init__(init_lr, model)
        self.lr = init_lr
        self.momentum = momentum
        # Initialize velocity for momentum (one for each layer parameter)
        self.velocity = {}
        for layer, param_name, param, grad in self.model.parameters():
            self.velocity[(layer, param_name)] = cp.zeros_like(param)

    def step(self):
        """
        Performs a single optimization step, updating model parameters using SGD with momentum.
        """
        for layer, param_name, param, grad in self.model.parameters():
            if grad is None:
                continue  # Skip parameters with no gradient
            # Update velocity with momentum
            key = (layer, param_name)
            self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
            # Update parameter using velocity
            param += self.velocity[key]


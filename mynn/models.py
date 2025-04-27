from .op import *
import pickle
import cupy as cp
from typing import *

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers:List[Linear] = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_features=size_list[i], out_features=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                elif act_func == 'LeakyReLU':
                    layer_f = LeakyReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                for param_name, param, grad in layer.parameters():
                    params.append((layer, param_name, param, grad))
        return params
    
    def load_model(self, param_path: str):
        """
        Load model parameters from a pickle file.
        
        Args:
            param_path (str): Path to the pickle file containing model parameters
        """
        with open(param_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.size_list = saved_data['size_list']
        self.act_func = saved_data['act_func']
        self.layers = []
        
        # Reconstruct layers
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_features=self.size_list[i], out_features=self.size_list[i + 1])
            # Load saved weights and biases
            layer.W = cp.array(saved_data['layers'][2*i]['W'])
            layer.b = cp.array(saved_data['layers'][2*i]['b'])
            
            if saved_data['layers'][2*i].get('weight_decay', False):
                layer.weight_decay = True
                layer.weight_decay_lambda = saved_data['layers'][2*i]['weight_decay_lambda']
                
            self.layers.append(layer)
            
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    self.layers.append(Logistic())
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'LeakyReLU':
                    self.layers.append(LeakyReLU())

    def save_model(self, save_path: str):
        """
        Save model parameters to a pickle file.
        
        Args:
            save_path (str): Path where the model parameters will be saved
        """
        model_data = {
            'size_list': self.size_list,
            'act_func': self.act_func,
            'layers': []
        }
        
        # Save parameters for each linear layer
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer_data = {
                    'W': layer.W.get(),  # Convert CuPy array to numpy for pickling
                    'b': layer.b.get(),
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda
                }
                model_data['layers'].append(layer_data)
            else:
                model_data['layers'].append(None)  # Placeholder for activation layers
                
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        pass

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, loss_grad):
        pass
    
    def load_model(self, param_list):
        pass
        
    def save_model(self, save_path):
        pass
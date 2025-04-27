# import numpy as np
# import cupy as cp
# import os
# from tqdm import tqdm
# from .op import Linear, ReLU

# class RunnerM():
#     """
#     This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
#     due to the different implementation of those models.
#     """
#     def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
#         self.model = model
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.metric = metric
#         self.scheduler = scheduler
#         self.batch_size = batch_size

#         self.train_scores = []
#         self.dev_scores = []
#         self.train_loss = []
#         self.dev_loss = []

#     def train(self, train_set, dev_set, **kwargs):

#         num_epochs = kwargs.get("num_epochs", 0)
#         log_iters = kwargs.get("log_iters", 100)
#         save_dir = kwargs.get("save_dir", "best_model")

#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)

#         best_score = 0

#         for epoch in tqdm(range(num_epochs)):
#             X, y = train_set

#             assert X.shape[0] == y.shape[0]

#             idx = np.random.permutation(range(X.shape[0]))

#             X = X[idx]
#             y = y[idx]

#             global_initial_W = {id(layer): layer.W.copy() for layer in self.model.layers if isinstance(layer, Linear)}
#             param_ids = {id(layer): (id(layer.W), id(layer.b)) for layer in self.model.layers if isinstance(layer, Linear)}
#             last_W = None  # 记录上一次迭代更新后的 W
           
#             for iteration in range(int(X.shape[0] / self.batch_size) + 1):
#                 train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
#                 train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

#                 logits = self.model(train_X)
#                 trn_loss = self.loss_fn(logits, train_y)
#                 self.train_loss.append(trn_loss)
                
#                 trn_score = self.metric(logits, train_y)
#                 self.train_scores.append(trn_score)

#                 # the loss_fn layer will propagate the gradients.
#                 self.loss_fn.backward()

#                 self.optimizer.step()
#                 if self.scheduler is not None:
#                     self.scheduler.step()
                
#                 dev_score, dev_loss = self.evaluate(dev_set)
#                 self.dev_scores.append(dev_score)
#                 self.dev_loss.append(dev_loss)

#                 if (iteration) % log_iters == 0:
#                     print(f"epoch: {epoch}, iteration: {iteration}")
#                     print(f"[Train] loss: {trn_loss}, score: {trn_score}")
#                     print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

#             if dev_score > best_score:
#                 save_path = os.path.join(save_dir, 'best_model.pickle')
#                 self.save_model(save_path)
#                 print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
#                 best_score = dev_score
#         self.best_score = best_score

#     def evaluate(self, data_set):
#         X, y = data_set
#         logits = self.model(X)
#         loss = self.loss_fn(logits, y)
#         score = self.metric(logits, y)
#         return score, loss
    
#     def save_model(self, save_path):
#         self.model.save_model(save_path)

import numpy as np
import cupy as cp
import os
from tqdm import tqdm
from .op import Linear, ReLU

class RunnerM:
    """
    This is an example to train, evaluate, save, load the model. Modified to include debugging logs and data standardization
    to address training issues (e.g., loss stuck at ~2.3).
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        best_score = 0
        X, y = train_set

        # 标准化数据
        X = (X - cp.mean(X)) / cp.std(X)
        # print("Train X min:", float(cp.min(X)), "max:", float(cp.max(X)), 
        #       "mean:", float(cp.mean(X)), "std:", float(cp.std(X)))
        
        # 验证集标准化
        dev_X, dev_y = dev_set
        dev_X = (dev_X - cp.mean(dev_X)) / cp.std(dev_X)
        # print("Dev X min:", float(cp.min(dev_X)), "max:", float(cp.max(dev_X)), 
        #       "mean:", float(cp.mean(dev_X)), "std:", float(cp.std(dev_X)))
        dev_set = (dev_X, dev_y)

        for epoch in tqdm(range(num_epochs)):
            assert X.shape[0] == y.shape[0]
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            # print(f"Epoch {epoch}, Learning rate: {self.optimizer.lr}, Momentum: {self.optimizer.momentum}")

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                global_iter = epoch * (int(X.shape[0] / self.batch_size) + 1) + iteration
                train_X = X[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                # print(f"Iteration {global_iter}, Batch size: {train_X.shape[0]}")

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                assert trn_loss != np.nan

                self.train_loss.append(float(trn_loss))
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()

                self.optimizer.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(float(dev_loss))

                if global_iter % log_iters == 0:
                    print(f"Epoch: {epoch}, Iteration: {global_iter}")
                    print(f"[Train] loss: {float(trn_loss)}, score: {trn_score}")
                    print(f"[Dev] loss: {float(dev_loss)}, score: {dev_score}")

                if self.scheduler is not None:
                    self.scheduler.step()

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"Best accuracy performance updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score
        print(f"Best score: {self.best_score}")

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, float(loss)
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
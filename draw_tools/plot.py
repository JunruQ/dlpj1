import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]
    
    # 将 cupy 数组转换为 numpy 数组
    train_loss = cp.asnumpy(runner.train_loss) if isinstance(runner.train_loss, cp.ndarray) else np.array(runner.train_loss)
    dev_loss = cp.asnumpy(runner.dev_loss) if isinstance(runner.dev_loss, cp.ndarray) else np.array(runner.dev_loss)
    train_scores = cp.asnumpy(cp.array(runner.train_scores)) if isinstance(runner.train_scores[0], cp.ndarray) else np.array(runner.train_scores)
    dev_scores = cp.asnumpy(cp.array(runner.dev_scores)) if isinstance(runner.dev_scores[0], cp.ndarray) else np.array(runner.dev_scores)
    
    epochs = np.arange(len(train_scores))
    
    # 绘制训练损失变化曲线
    axes[0].plot(epochs, train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(epochs, dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')
    
    # 绘制训练准确率变化曲线
    axes[1].plot(epochs, train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(epochs, dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')
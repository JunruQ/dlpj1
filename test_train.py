import mynn as nn
from draw_tools.plot import plot
import cupy as cp
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

cp.random.seed(309)

train_images_path = 'dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = 'dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = cp.frombuffer(f.read(), dtype=cp.uint8)

idx = cp.random.permutation(cp.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(cp.asnumpy(idx), f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 初始化模型
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 512, 10], 'LeakyReLU', [1e-5, 1e-5])
optimizer = nn.optimizer.SGD(init_lr=0.05, model=linear_model, momentum=0.9)
scheduler = nn.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, num_classes=int(train_labs.max()) + 1)

# 初始化 runner
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, batch_size=256)

# 训练
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=200, log_iters=10000, save_dir=r'./saved_models')

# 绘图
f, axes = plt.subplots(1, 2)
f.set_size_inches(8, 4)
axes = axes.ravel()
f.set_tight_layout(True)
plot(runner, axes)

plt.savefig('figs/train.png', dpi=300, bbox_inches='tight')
# plt.show()
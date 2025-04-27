# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP([28*28, 512, 10], 'LeakyReLU', [1e-5, 1e-5])
model.load_model('best_models/best_model.pickle')

test_images_path = 'dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# Weight visualization
def visualize_weights(model, num_filters=16, grid_size=(4, 4)):
    """
    Visualize the weights of the first layer as 28x28 images.
    
    Args:
        model: Model_MLP instance
        num_filters: Number of weight vectors to visualize
        grid_size: Tuple of (rows, cols) for the visualization grid
    """
    # Get the weights of the first Linear layer
    first_layer = next(layer for layer in model.layers if isinstance(layer, nn.models.Linear))
    weights = first_layer.W.get()  # Convert CuPy to numpy
    
    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 6))
    axes = axes.ravel()
    
    # Normalize weights for visualization
    for i in range(min(num_filters, weights.shape[1])):
        # Reshape weights to 28x28
        weight_img = weights[:, i].reshape(28, 28)
        
        # Normalize for better visualization
        weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min())
        
        # Display
        axes[i].imshow(weight_img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    plt.tight_layout()
    plt.savefig('weight_visualization.png')
    plt.close()

visualize_weights(model, num_filters=32, grid_size=(4, 8))
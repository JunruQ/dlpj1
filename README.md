# Deep Learning Project 1

This project builds a simple neural network framework from scratch and applies it to the MNIST dataset. Main functionalities include:

- Dataset exploration and visualization
- Neural network model construction
- Training curve plotting
- Weight visualization
- Hyperparameter search (learning rate, activation function, optimizer, etc.)
- Model saving and loading

## Project Structure

```
|-- best_models/                # Best trained models
|-- dataset/                    # Original MNIST dataset
|-- draw_tools/                 # Tools for plotting curves and weights
|-- figs/                       # Generated figures during training
|-- mynn/                       # Custom neural network framework
|-- saved_models/               # Other saved models
|-- dataset_explore.ipynb        # Dataset exploration notebook
|-- hyperparameter_search.py    # Hyperparameter search script
|-- test_model.py                # Model testing script
|-- test_train.py                # Model training script
|-- weight_visualization.py      # Weight visualization script
|-- idx.pickle                   # Auxiliary index file
|-- README.md                    # Project documentation
```

## Quick Start

### Clone the repository

```bash
git clone https://github.com/JunruQ/dlpj1.git
cd dlpj1
```

### Download the trained model

[Click here to download `best_model.pickle`](https://drive.google.com/file/d/1yY3J7IjH1q2S4V2NG_sEgkFQWZIkSFBd)

Place it under the `best_models/` directory.

### Install dependencies

The project mainly uses standard Python libraries. (If you want to run Jupyter notebooks, make sure to install `jupyter`.)

### Run Examples

Train a new model:
```bash
python test_train.py
```

Test the best saved model:
```bash
python test_model.py
```

Perform hyperparameter search:
```bash
python hyperparameter_search.py
```

Visualize the model weights:
```bash
python weight_visualization.py
```

## Main Features

- **Training Curves**: Plot loss and accuracy curves across epochs, saved under `figs/`
- **Activation Function Comparison**: Compare performance of ReLU, LeakyReLU, etc.
- **Learning Rate Scheduler Comparison**: Support for StepLR, MultiStepLR, ExponentialLR
- **Weight Visualization**: Visualize the weights of the first layer
- **Model Saving**: Save the model with the best validation performance

## Contact

For questions or suggestions, feel free to open an [issue](https://github.com/JunruQ/dlpj1/issues)!


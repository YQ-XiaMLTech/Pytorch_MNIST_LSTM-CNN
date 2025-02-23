# LSTM Classification on MNIST

This project utilizes **LSTM (Long Short-Term Memory Network)** for multi-class classification on the MNIST handwritten digit dataset. This document provides a brief overview of the **Structure of the Repository, Steps to Reproduce the Results, Model Architecture, Hyperparameter Tuning Methods, Results, and Techniques to Prevent Overfitting**.

---
## 1. Structure of the Repository
```
MNIST_LSTM
┌── data
├── logs
├── saves
├── environment.yml
├── README.md
├── mnist_lstm.py
├── mnist_lstm_optuna.py
├── model.py
└── utils.py
```

- **`data/`**: Folder that contains the MNIST dataset.  
- **`logs/`**: Folder that contains log files.  
- **`saves/`**: Folder that contains saved models and training record figures.  
- **`environment.yml`**: A Conda environment file describing the Python packages and versions needed to reproduce the environment.  
- **`README.md`**: The README file, providing instructions and information about the project.  
- **`mnist_lstm.py`**: The primary script for training and testing the LSTM model on the MNIST dataset.  
- **`mnist_lstm_optuna.py`**: A script integrating Optuna for hyperparameter tuning on the LSTM model.  
- **`model.py`**: The script defining the LSTM model architecture.  
- **`utils.py`**: Utility scripts, such as custom data loaders, logging functions, or other helper routines.

---  
## 2. Steps to Reproduce the Results

1. **Download the repository**  
2. **Create and activate the Conda environment**  
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```
3. **Run the LSTM training script**  
   ```bash
   python mnist_lstm.py
   ```
   - This script automatically downloads the MNIST dataset and stores it in the `data/` folder, trains the model with early stopping, and saves training logs in the `logs/` folder.  
   - The trained model and related training curves are stored in the `saves/` folder, which includes the following files:
     - `accuracies.png`: Training and validation accuracy plot.  
     - `errors.png`: Training and validation error plot.  
     - `loss_train_and_val.png`: Training and validation loss curve.  
     - `final_state_dict.pth`: Final model state dictionary.  
     - `model_last.pt`: The last saved model checkpoint.  

4. **Run Optuna hyperparameter search**  
   ```bash
   python mnist_lstm_optuna.py
   ```
   - This script performs multiple trials to search for the best hyperparameters, outputs the optimal results, and generates related visualizations (e.g., `optimization_history.png`).

## 3. Model Architecture

This project employs an **LSTM** model to process the 2D MNIST images (28×28 pixels) by treating them as sequential inputs:

1. **Input Layer**: Each image is reshaped to `(batch_size, 28, 28)`, where each time step corresponds to one row of 28 pixels.  
2. **LSTM Layer**: Includes `num_layers` (default is `3`) with a tunable hidden size (`hidden_size`, default `95`) and `dropout=0.47`.  
3. **Fully Connected Layer (FC or Dense Layer)**: Maps the output from the last time step of the LSTM to 10 classes (one for each digit).  
4. **Softmax / log_softmax**: In this project, `log_softmax` is used in the forward pass, paired with `NLLLoss` during training for multi-class classification.

A simplified structure diagram is shown below:

```
Input (batch, 28, 28)
  --> LSTM (2-layer, hidden_size=62, dropout=0.2)
    --> Output (last time step)
      --> Fully Connected (hidden_size -> 10)
        --> log_softmax
          --> NLLLoss (labels in integer form)
```
---

## 4. Hyperparameter Tuning Methods

### 1. **Script Arguments (`argparse`)**  
   - **Learning rate** (`--lr`): Can be manually specified (default: `0.001`).
   - **Weight decay** (`--weight-decay`): Default is `5.36e-06`.
   - **Batch size** (`--batch-size`): Default is `96`.  
   - **LSTM Parameters**:
     - `--input_size=28` (fixed)  
     - `--hidden_size` (default: `95`)  
     - `--num_layers` (default: `3`)  
     - `--dropout` (default: `0.47`)  

### 2. **Optuna Hyperparameter Optimization**  
   - The script `mnist_lstm_optuna.py` contains the `objective()` function, which automatically searches for optimal hyperparameters, including:
     - Optimizer (Adam / SGD)  
     - Learning rate (`1e-5` to `1e-1`, log scale)
     - Weight decay (`1e-6` to `1e-2`, log scale)
     - LSTM hidden size (`32` to `128`)  
     - Number of LSTM layers (`1` to `3`)  
     - Dropout rate (`0.1` to `0.5`)  
     - Batch size (`64` to `256`, step size of 64)  

Optuna performs multiple trials to explore different hyperparameter configurations, returning the **best validation accuracy** along with the corresponding hyperparameters. Additionally, it generates various visualizations, including the **optimization history plot, contour plot, parallel coordinate plot, and parameter importance plot**.

---

## 5. Results

With the default configuration (`hidden_size=95, num_layers=3, dropout=0.47, lr=0.001, batch_size=96, weight_decay=5.36e-06`), after training for 30 epochs (with early stopping enabled), the model generally achieves high accuracy on the MNIST dataset. The training results (which may vary slightly due to random seed or environment differences) are as follows:

- **Training set**: Accuracy ~ 99.6875%, Error ~ 0.3125%  
- **Validation set**: Accuracy ~ 98.9667%, Error ~ 1.0333%  
- **Test set**: Accuracy ~ 98.6500%, Error ~ 1.3500%  

The log file in `logs/record.txt` contains details of each epoch, including loss, accuracy, error rates, and time-related statistics such as **Time, Time total, and Time remain**.  

Training process visualizations can be found in `saves/YYYYMMDD_HHMMSS_f/`, including the following files:  
- `accuracies.png`: Accuracy curves  
- `errors.png`: Error curves  
- `loss_train_and_val.png`: Training and validation loss curves  
- `final_state_dict.pth`: The final model state dictionary  
- `model_last.pt`: The last saved model checkpoint  

---

## 6. Techniques to Prevent Overfitting

1. **Dropout**: Applied dropout of `0.47` in LSTM layers to randomly drop some neuron connections, reducing overfitting.  
2. **Early Stopping**: If the validation loss does not improve for `patience` consecutive epochs, training stops early (default `patience=10`).  
3. **Optimization of Hidden Layer Dimensions**: Properly tuning `hidden_size`, along with L2 regularization and batch size adjustments, helps mitigate overfitting.  


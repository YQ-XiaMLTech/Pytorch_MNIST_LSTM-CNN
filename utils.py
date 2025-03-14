import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

def batch_generator(dataset, batch_size, shuffle=True):
    """
    Custom batch generator for iterating over an MNIST dataset.

    This function generates batches of data from the given dataset, allowing for
    shuffling to randomize the order of samples within an epoch.

    Args:
        dataset (torch.utils.data.Dataset): The MNIST dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the dataset at the beginning of each epoch. Defaults to True.

   Returns:
        tuple: A batch of data in the form `(batch_x, batch_y)`, where:
               - batch_x (numpy.ndarray): Array of image tensors converted to NumPy format.
               - batch_y (numpy.ndarray): Corresponding labels for the batch.
    """
    # Get dataset size
    data_size = len(dataset)
    # Generate index array
    indices = np.arange(data_size)

    # Shuffle indices if required
    if shuffle:
        np.random.shuffle(indices)

    batch_count = 0
    while True:
        # If the epoch is completed, reshuffle and restart batch count
        if batch_count * batch_size >= data_size:
            batch_count = 0
            if shuffle:
                np.random.shuffle(indices)

        # Determine start and end indices of the batch
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1

        # Get batch indices
        batch_indices = indices[start:end]

        # Retrieve batch data using indices
        batch_data = [dataset[i] for i in batch_indices]
        batch_x = np.array([data[0].numpy() for data in batch_data])
        batch_y = np.array([data[1] for data in batch_data])

        yield batch_x, batch_y



def train(model, device, train_set, optimizer, criterion, batch_size):
    """
    Train the LSTM model on the MNIST dataset.

    This function iterates through the training set using a batch generator,
    computes loss and accuracy, updates model weights, and returns statistics
    for performance evaluation.

    Args:
        model (torch.nn.Module): The LSTM model.
        device (torch.device): The device (CPU or GPU) to run training on.
        train_set (torch.utils.data.Dataset): The training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function (NLLLoss).
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: Updated model, average training loss, training accuracy, and error rate.
    """
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    num_samples = len(train_set)
    train_gen = batch_generator(train_set, batch_size, shuffle=True)  # Generate training batches
    num_batches = num_samples // batch_size
    for batch_idx in range(num_batches):
        batch_x, batch_y = next(train_gen)
        # Convert NumPy arrays to PyTorch tensors
        data = torch.tensor(batch_x, dtype=torch.float32, device=device)
        target = torch.tensor(batch_y, dtype=torch.long, device=device)
        optimizer.zero_grad()  # Reset gradients

        data = data.squeeze()  # Remove unnecessary dimensions
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        pred = output.argmax(dim=1, keepdim=True)  # Get predicted labels
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item() * data.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / num_samples
    accuracy = 100.0 * correct / num_samples
    error = 100.0 - accuracy
    return model, avg_loss, accuracy, error


def test(model, device, test_set, criterion, batch_size):
    """
    Evaluate the LSTM model on the test dataset.

    This function iterates through the test set using a batch generator,
    computes loss and accuracy, and returns performance statistics.

    Args:
        model (torch.nn.Module): The trained LSTM model.
        device (torch.device): The device (CPU or GPU) to run evaluation on.
        test_set (torch.utils.data.Dataset): The test dataset.
        criterion (torch.nn.Module): Loss function (NLLLoss).
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: Model, average test loss, test accuracy, and error rate.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    num_samples = len(test_set)
    test_gen = batch_generator(test_set, batch_size, shuffle=False)  # Generate test batches
    num_batches = num_samples // batch_size
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for batch_idx in range(num_batches):
            batch_x, batch_y = next(test_gen)
            data = torch.tensor(batch_x, dtype=torch.float32, device=device)
            target = torch.tensor(batch_y, dtype=torch.long, device=device)

            data = data.squeeze()
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.item() * data.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / num_samples
    accuracy = 100.0 * correct / num_samples
    error = 100.0 - accuracy
    return model, avg_loss, accuracy, error

def get_timestring():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
       Draw a 2D plot of several lines
       :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
       :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
       :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
       :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
       :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
       :param legend_fontsize: (float) legend fontsize. e.g., 15
       :param fig_title: (string) title of the figure. e.g., "Anonymous"
       :param fig_x_label: (string) x label of the figure. e.g., "time"
       :param fig_y_label: (string) y label of the figure. e.g., "val"
       :param show_flag: (boolean) whether you want to show the figure. e.g., True
       :param save_flag: (boolean) whether you want to save the figure. e.g., False
       :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
       :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
       :param fig_title_size: (float) figure title size. e.g., 20
       :param fig_grid: (boolean) whether you want to display the grid. e.g., True
       :param marker_size: (float) marker size. e.g., 0
       :param line_width: (float) line width. e.g., 1
       :param x_label_size: (float) x label size. e.g., 15
       :param y_label_size: (float) y label size. e.g., 15
       :param number_label_size: (float) number label size. e.g., 15
       :param fig_size: (tuple) figure size. e.g., (8, 6)
       :return:
       """
    assert len(y_lists[0]) == len(x_list), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i],
               linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()
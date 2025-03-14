import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_LSTM(nn.Module):
    """
    LSTM model for MNIST digit classification.

    This model processes MNIST images as sequences of pixels and classifies
    them into one of 10 digit categories using an LSTM-based architecture.

    Args:
        input_size (int): Number of input features per time step (28 for MNIST).
        hidden_size (int): Number of hidden units in the LSTM.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate applied between LSTM layers.
        num_classes (int, optional): Number of output classes (default: 10).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=10):
        super(MNIST_LSTM, self).__init__()
        # Define the LSTM layer(s)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        # Fully connected layer mapping LSTM output to 10 digit classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size),
                              where sequence_length = 28 (rows or columns of the image).

        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, num_classes).
        """
        output, _ = self.lstm(x, None)
        out = self.fc(output[:, -1, :])
        # Apply log softmax to get log probabilities for classification
        return F.log_softmax(out, dim=1)



import argparse
import torch
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms
import os.path as osp
import os
from model import *
from utils import *


def mnist_lstm():
    """
    This function trains an LSTM model on the MNIST dataset for handwritten digit classification.
    It handles data loading, model initialization, training, validation, testing, logging,
    and saving the final model and results.
    """

    # Parse Command-line Arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch-size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 96)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', default=5.36e-06, type=float, metavar='W',
                        help='weight-decay (default: 5.36e-06)')
    parser.add_argument('--device_type', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='Specify the device type: "gpu" or "cpu" (default: "cpu")')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--input_size", type=int, default=28, help="Input size for LSTM (28 for MNIST)")
    parser.add_argument("--hidden_size", type=int, default=95, help="Hidden size of LSTM")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.47, help="Dropout rate in LSTM")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device_type.lower() == "gpu" else "cpu")
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # 1. Prepare Dataset
    print("[Step 1] Preparing dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Download and load MNIST dataset
    full_train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Split into training and validation sets
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_set,
    #                                            batch_size=batch_size,
    #                                            shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set,
    #                                          batch_size=batch_size,
    #                                          shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 2. Setup Save Paths
    print("[Step 2] Save path")
    timestring = get_timestring()
    main_path = ''
    main_save_path = osp.join(main_path, "saves", timestring)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    model_restart_path = osp.join(main_save_path, "model_restart.pt")
    final_model_state_path = osp.join(main_save_path, "final_state_dict.pth")
    figure_save_path_train_loss_whole = osp.join(main_save_path, "loss_train.png")
    figure_save_path_combined = osp.join(main_save_path, "loss_train_and_val.png")
    figure_save_path_accuracies_whole = osp.join(main_save_path, "accuracies.png")
    figure_save_path_errors_whole = osp.join(main_save_path, "errors.png")
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_loss_train_whole: {}".format(figure_save_path_train_loss_whole))
    print("figure_save_path_combined: {}".format(figure_save_path_combined))
    print("figure_save_path_accuracies_whole: {}".format(figure_save_path_accuracies_whole))
    print("figure_save_path_errors_whole: {}".format(figure_save_path_errors_whole))

    log_dir = osp.join(main_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = osp.join(log_dir, "record.txt")
    with open(log_file_path, "w") as f:
        f.write("")

    # 3. Initialize Model
    print("[Step 3] Initializing model")
    model = MNIST_LSTM(args.input_size, args.hidden_size, args.num_layers, args.dropout).to(device)
    # Set optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()  # Set loss function

    # Load a checkpoint if available
    if os.path.exists(model_restart_path):
        checkpoint = torch.load(model_restart_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 4. Train the Model
    print("[Step 4] Training...")
    start_time = time.time()
    start_time_0 = start_time

    epoch_train_loss_list = []
    epoch_val_loss_list = []
    lr_list = []
    epoch_accuracies_list_train = []
    epoch_accuracies_list_val = []
    epoch_errors_list_train = []
    epoch_errors_list_val = []

    patience = 10  # Early stopping patience
    min_delta = 1e-4  # Minimum improvement required for early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state_dict = None

    for epoch in range(1, args.epochs + 1):
        # Train and validate for one epoch
        model, train_loss, accuracies_train, errors_train = train(model, device, train_set, optimizer, criterion,
                                                                  args.batch_size)
        model, val_loss, accuracies_val, errors_val = test(model, device, val_set, criterion, args.batch_size)
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(val_loss)
        lr_list.append(optimizer.param_groups[0]["lr"])

        epoch_accuracies_list_train.append(accuracies_train)
        epoch_accuracies_list_val.append(accuracies_val)
        epoch_errors_list_train.append(errors_train)
        epoch_errors_list_val.append(errors_val)

        # Save best model based on validation loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset early stopping counter
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, model_save_path)
        else:
            epochs_no_improve += 1

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.6f}")
            break

        now_time = time.time()
        test_log = (
            f"Epoch [{epoch:05d}/{args.epochs:05d}] "
            f"Loss_train: {train_loss:.6f} "
            f"Loss_val: {val_loss:.6f} "
            f"Accuracies_train: {accuracies_train:.4f}, errors_train: {errors_train:.4f} "
            f"Accuracies_val: {accuracies_val:.4f}, errors_val: {errors_val:.4f} "
            f"Lr: {optimizer.param_groups[0]['lr']:.6f} "
            f"(Time: {now_time - start_time:.6f}s "
            f"Time total: {(now_time - start_time_0) / 60.0:.2f}min "
            f"Time remain: {(now_time - start_time_0) / 60.0 / epoch * (args.epochs - epoch):.2f}min)"
        )
        print(test_log)
        with open(log_file_path, "a") as f:
            f.write(test_log + "\n")
        start_time = now_time

    # Load the best model
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        print("Loaded best model from epoch with lowest val_loss.")

    # 5. Evaluate on Test Set
    print("[Step 5] Test...")
    # Evaluate model on all three datasets
    model, test_loss_traindata, accuracies_test_traindata, errors_test_traindata = test(model, device, train_set,
                                                                                        criterion, args.batch_size)
    model, test_loss_valdata, accuracies_test_valdata, errors_test_valdata = test(model, device, val_set, criterion,
                                                                                  args.batch_size)
    model, test_loss_testdata, accuracies_test_testdata, errors_test_testdata = test(model, device, test_set, criterion,
                                                                                     args.batch_size)

    test_log = (
        f"Test Results_traindata:\n"
        f"Loss: {test_loss_traindata:.6f}\n"
        f"accuracies_traindata: {accuracies_test_traindata:.4f}, errors_traindata: {errors_test_traindata:.4f}\n"
        f"Test Results_valdata:\n"
        f"Loss: {test_loss_valdata:.6f}\n"
        f"accuracies_valdata: {accuracies_test_valdata:.4f}, errors_valdata: {errors_test_valdata:.4f}\n"
        f"Test Results_testdata:\n"
        f"Loss: {test_loss_testdata:.6f}\n"
        f"accuracies_testdata: {accuracies_test_testdata:.4f}, errors_testdata: {errors_test_testdata:.4f}\n"
    )
    print(test_log)
    with open(log_file_path, "a") as f:
        f.write(test_log + "\n")

    # 6. Drawing training result
    print("[Step 6] Drawing training result...")
    loss_length = len(epoch_train_loss_list)
    loss_x = range(1, loss_length + 1)

    # draw train and validation loss
    draw_two_dimension(
        y_lists=[epoch_train_loss_list, epoch_val_loss_list],
        x_list=loss_x,
        color_list=["blue", "red"],
        legend_list=["Train Loss", "Validation Loss"],
        line_style_list=["solid", "dashed"],
        fig_title="Train and Validation loss",
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_combined
    )

    # draw train loss_whole
    draw_two_dimension(
        y_lists=[epoch_train_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_train_loss_list[-1], min(epoch_train_loss_list))],
        line_style_list=["solid"],
        fig_title="Train loss - whole",
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_train_loss_whole
    )
    # draw train and validation accuracies
    draw_two_dimension(
        y_lists=[epoch_accuracies_list_train, epoch_accuracies_list_val],
        x_list=list(range(1, len(epoch_accuracies_list_train) + 1)),
        color_list=["green", "orange"],
        line_style_list=["solid", "dashed"],
        legend_list=[
            "Train Acc: last={0:.4f}, max={1:.4f}".format(
                epoch_accuracies_list_train[-1],
                max(epoch_accuracies_list_train)
            ),
            "Val Acc:   last={0:.4f}, max={1:.4f}".format(
                epoch_accuracies_list_val[-1],
                max(epoch_accuracies_list_val)
            ),
        ],
        fig_title="Accuracy - Whole",
        fig_x_label="Epoch",
        fig_y_label="Accuracy",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_accuracies_whole,
    )
    # draw train and validation errors
    draw_two_dimension(
        y_lists=[epoch_errors_list_train, epoch_errors_list_val],
        x_list=list(range(1, len(epoch_errors_list_train) + 1)),
        color_list=["green", "orange"],
        line_style_list=["solid", "dashed"],
        legend_list=[
            "Train Acc: last={0:.4f}, max={1:.4f}".format(
                epoch_errors_list_train[-1],
                max(epoch_errors_list_train)
            ),
            "Val Acc:   last={0:.4f}, max={1:.4f}".format(
                epoch_errors_list_val[-1],
                max(epoch_errors_list_val)
            ),
        ],
        fig_title="errors - Whole",
        fig_x_label="Epoch",
        fig_y_label="errors",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_errors_whole,
    )

    # 7. Save the Final Model
    print("[Step 7] Saving final model...")
    final_state_dict = model.state_dict()
    torch.save(final_state_dict, final_model_state_path)


if __name__ == '__main__':
    mnist_lstm()

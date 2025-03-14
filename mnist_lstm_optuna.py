import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from model import *
from utils import *
import optuna

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    This function defines the search space for hyperparameters and evaluates different
    configurations by training an LSTM model on the MNIST dataset. The goal is to
    maximize validation accuracy.

    Args:
        trial (optuna.Trial): An Optuna trial object for hyperparameter sampling.

    Returns:
        float: The best validation accuracy obtained with the selected hyperparameters.
    """
    # Set training parameters
    EPOCHS = 30
    seed=1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure reproducibility by setting random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # Define the hyperparameter search space
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    print("[Step 1] Preparing dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load MNIST dataset
    full_train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print("[Step 3] Initializing model")
    model = MNIST_LSTM(28, hidden_size, num_layers, dropout).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    print("[Step 4] Training...")
    best_accuracy = 0.0
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        num_samples = len(train_set)
        train_gen = batch_generator(train_set, batch_size, shuffle=True)
        num_batches = num_samples // batch_size
        for batch_idx in range(num_batches):
            batch_x, batch_y = next(train_gen)
            data = torch.tensor(batch_x, dtype=torch.float32, device=device)
            target = torch.tensor(batch_y, dtype=torch.long, device=device)
            optimizer.zero_grad()

            data = data.squeeze()  # Ensure correct shape for LSTM
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        total_loss = 0.0
        correct = 0
        num_samples_val = len(val_set)
        test_gen = batch_generator(val_set, batch_size, shuffle=False)
        num_batches = num_samples_val // batch_size
        with torch.no_grad():
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

        # Compute validation accuracy
        accuracy = 100.0 * correct / num_samples_val
        print("accuracy = ", accuracy)
        # Track the best validation accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        # Report accuracy to Optuna for pruning decisions
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_accuracy

def main():
    """
    Main function to run Optuna hyperparameter tuning.

    This function creates an Optuna study, performs multiple trials to optimize
    hyperparameters, and visualizes the results with various plots.
    """
    # Create an Optuna study to maximize validation accuracy
    study = optuna.create_study(direction='maximize')
    # Run the optimization with 50 trials
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameter results
    best_trial = study.best_trial
    print(f"\n===== Done! Best trial: {best_trial.number} =====")
    print(f"  Val Accuracy = {best_trial.value:.4f}")
    print("  Best params:", best_trial.params)

    # Generate and save visualizations for analysis
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image("optuna/optimization_history.png")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image("optuna/param_importances.png")

    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.write_image("optuna/parallel_coordinate.png")

    fig4 = optuna.visualization.plot_contour(study, params=['batch_size', 'lr'])
    fig4.write_image("optuna/contour_plot.png")

if __name__ == "__main__":
    main()





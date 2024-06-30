import torch
import optuna
from model.model import CNNModel
import os


def save_model(model, filename):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    filepath = os.path.join('saved_models', filename)
    torch.save(model.state_dict(), filepath)

def load_model(filename, model_params=None):
    filepath = os.path.join('saved_models', filename)
    model = CNNModel() if model_params is None else CNNModel(**model_params)
    model.load_state_dict(torch.load(filepath))
    return model

def objective(trial, train_loader, val_loader):
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
    conv_filters = [trial.suggest_int(f"conv_filters_{i}", 16, 128) for i in range(num_conv_layers)]
    kernel_sizes = [trial.suggest_int(f"kernel_size_{i}", 2, 5) for i in range(num_conv_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_sizes = [trial.suggest_int(f"fc_size_{i}", 16, 128) for i in range(2)]
    
    input_channels = 3  # Make sure this matches the actual input shape of your data
    model = CNNModel(num_conv_layers=num_conv_layers, conv_filters=conv_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, fc_sizes=fc_sizes, input_channels=input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True))
    criterion = torch.nn.MSELoss()

    for epoch in range(10):  # Simplified for tuning purposes
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    if val_loader:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        return val_loss
    return loss.item()

def hyperparameter_optimization(train_loader, val_loader=None):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=20)
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.values}")
    print(f"Best hyperparameters: {best_trial.params}")
    return best_trial.params
import torch
import optuna
from model.model import CNNModel
import os

def save_model(model, model_params, filename):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    filepath = os.path.join('saved_models', filename)
    torch.save({'model_state_dict': model.state_dict(), 'model_params': model_params}, filepath)

def load_model(filename):
    filepath = os.path.join('saved_models', filename)
    checkpoint = torch.load(filepath)
    model_params = checkpoint['model_params']
    model = CNNModel(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_params

def calculate_output_size(input_size, kernel_size, stride, padding=0, dilation=1):
    return (input_size + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1

def objective(trial, train_loader, val_loader):
    num_conv_layers = 2  # Fixed to avoid issues
    conv_filters = [32, 64]  # Fixed to avoid issues
    kernel_sizes = [3, 3]  # Fixed to avoid issues
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_sizes = [64, 32]  # Fixed to avoid issues
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    input_channels = 3  # Ensure this matches the actual input shape of your data
    input_length = 3    # Length of the input sequence

    print(f"Checking parameters: num_conv_layers={num_conv_layers}, conv_filters={conv_filters}, kernel_sizes={kernel_sizes}, dropout_rate={dropout_rate}, fc_sizes={fc_sizes}, learning_rate={learning_rate}")

    # Calculate the output size after each convolution and pooling layer
    for i in range(num_conv_layers):
        input_length = calculate_output_size(input_length, kernel_sizes[i], stride=1, padding=1)
        print(f"After conv layer {i+1}, output size: {input_length}")
        if i < num_conv_layers - 1:  # Apply pooling only after the first convolution layer
            input_length = calculate_output_size(input_length, kernel_size=2, stride=2)  # Assuming max pooling with kernel size 2 and stride 2
            print(f"After pooling layer {i+1}, output size: {input_length}")
        if input_length <= 0:
            print(f"Pruning trial due to invalid output size after layer {i+1}.")
            raise optuna.exceptions.TrialPruned()

    # The output size of the final convolution layer
    fc_input_size = conv_filters[-1] * input_length
    print(f"Calculated fully connected input size: {fc_input_size}")

    model = CNNModel(num_conv_layers=num_conv_layers, conv_filters=conv_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, fc_sizes=fc_sizes, input_channels=input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(5):  # Simplified for tuning purposes
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Adjust targets shape to match outputs
            targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Training loss: {running_loss / len(train_loader)}")

    if val_loader:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                targets = targets.unsqueeze(1)  # Adjust targets shape to match outputs
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss}")
        return val_loss
    
    return running_loss / len(train_loader)

def hyperparameter_optimization(train_loader, val_loader=None):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=20)
    best_trial = study.best_trial
    print(f"Best trial value: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")
    return best_trial.params

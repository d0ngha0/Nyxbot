'''handle the path'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
'''computation and network'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from Model.esn import ESN,load_esn_models
from Model.mlp import MLP
'''handle the data'''
from Scripts.DataHandle import split_by_half_period, reshape_to_samples


def process_segments(models, data, period):
    """
    For each limb model, predict, split inputs and outputs.

    Parameters:
    - models: list of tuples (name, model, pred_slice, target_slice)
    - data: np.ndarray, full data_for_train
    - period: int

    Returns:
    - dict: mapping limb name to {in, out} arrays
    """
    results = {}
    for name, model, ps, ts in models:
        pred = model.predict(data[:, ps[0]:ps[1]])
        target = data[:, ts[0]:ts[1]]

        xin, _ = split_by_half_period(pred, period, 70, 70)
        _, xout = split_by_half_period(target, period, 70, 70)
        results[name] = {'in': xin, 'out': xout}

    return results

def reshape_all(results):
    reshaped = {}
    for limb, data in results.items():
        reshaped[limb] = {}
        for i, axis_name in enumerate(('z', 'y')):
            reshaped[limb][f'in_{axis_name}'] = reshape_to_samples(data['in'][:, i])
            reshaped[limb][f'out_{axis_name}'] = reshape_to_samples(data['out'][:, i])
    return reshaped

def mlp_train(data_path, model_path=None):
    # define the period of one moving circle
    period = 140
    # load the data for train the mlp
    data_for_train = np.load(data_path)
    esn_rf, esn_lf, esn_lh, esn_rh = load_esn_models('./Model/LearnedModel')

    print(data_for_train[90:data_for_train.shape[0]-50,:].shape[0]/period)
    models = [
    ("rf", esn_rf, (16, 20), (32, 34)),
    ("lf", esn_lf, (20, 24), (34, 36)),
    ("lh", esn_lh, (24, 28), (36, 38)),
    ("rh", esn_rh, (28, 32), (38, 40)),]

    mlp_data = process_segments(models, data_for_train[90:data_for_train.shape[0]-50,:], period)

    mlp_data_dic = reshape_all(mlp_data)

    model = train_best_mlp_model(mlp_data_dic['rh']['in_y'], mlp_data_dic['rh']['out_y'],  model_class=MLP)

    # Save the model state_dict
    if model_path is not None:
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model to {model_path}")
    
def train_best_mlp_model(in_array, out_array,  model_class, batch_size=16, lr=0.001, num_epochs=500, patience=20):
    """
    Trains an MLP model with early stopping on the provided data.

    Parameters:
    - in_array: np.ndarray, input features
    - out_array: np.ndarray, target outputs
    - model_path: str, where to save the best model (e.g., './Model/MLP_RH_Y.pth')
    - model_class: callable, a class that returns an uninitialized PyTorch MLP model
    - batch_size: int, batch size for training
    - lr: float, learning rate
    - num_epochs: int, max training epochs
    - patience: int, early stopping patience

    Returns:
    - model: the trained best model
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(in_array, dtype=torch.float32)
    Y_tensor = torch.tensor(out_array, dtype=torch.float32)

    # Train/val/test split
    X_temp, X_test, Y_temp, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.1765, random_state=42)

    # Create datasets and loaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model setup
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = float('inf')
    wait = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, Y_batch).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.6f}")
                break

    # Load and save best model
    model.load_state_dict(best_model_state)


    return model



if __name__ == '__main__':
    """
    Data format requirements for training:
    
    The data_for_train.npy file should contain a 2D array where:
    - Inputs (Joint Torques):
        RF (Right Front): columns 16-20
        LF (Left Front):  columns 20-24
        LH (Left Hind):   columns 24-28
        RH (Right Hind):  columns 28-32
    
    - Targets (Ground Reaction Forces - GRFs):
        RF (Right Front): columns 32-34
        LF (Left Front):  columns 34-36
        LH (Left Hind):   columns 36-38
        RH (Right Hind):  columns 38-40
    """
    # Replace data_path with your own data file path following the format specified above
    data_path = './DataForTrain/data_for_train.npy'
    model_path = "./Model/MLP_RH_Y.pth"
    mlp_train(data_path, model_path)
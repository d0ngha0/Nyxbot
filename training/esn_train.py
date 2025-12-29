
'''handle the path'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
'''computation and network'''
import numpy as np
from sklearn.metrics import mean_squared_error
from Model.esn import ESN
from pyswarms.single import GlobalBestPSO
'''PSO visualize'''
# from pyswarms.utils.plotters import plot_contour
# from pyswarms.utils.plotters.formatters import Mesher
'''handle the data'''
from Scripts.DataHandle import get_sample_ids, generate_data_sets
import pickle


def optimize_esn_with_pso(train_set, val_set, limb_str):
    """
    Optimize ESN hyperparameters using PSO for a specific limb.
    
    Parameters:
        train_set (np.ndarray): Training data array.
        val_set (np.ndarray): Validation data array.
        limb_str (str): Limb identifier, one of {"RF", "LF", "LH", "RH"}.
        
    Returns:
        best_cost (float): Best MSE from PSO optimization.
        best_pos (list): Best hyperparameter set [reservoir_size, sr, sparsity, leak_rate].
    """
    input_slices = {
        "RF": (16, 20),
        "LF": (20, 24),
        "LH": (24, 28),
        "RH": (28, 32)
    }
    target_slices = {
        "RF": (32, 34),
        "LF": (34, 36),
        "LH": (36, 38),
        "RH": (38, 40)
    }

    if limb_str not in input_slices or limb_str not in target_slices:
        raise ValueError(f"Invalid limb_str '{limb_str}'. Must be one of: {list(input_slices.keys())}")

    # Extract input and target slices
    in_start, in_end = input_slices[limb_str]
    out_start, out_end = target_slices[limb_str]

    train_inputs = train_set[:, in_start:in_end]
    train_targets = train_set[:, out_start:out_end]
    val_inputs = val_set[:, in_start:in_end]
    val_targets = val_set[:, out_start:out_end]

    # Define hyperparameter bounds: reservoir_size, sr, sparsity, leak_rate
    # bounds 1
    # bounds = (
    #     [60,  0.8,  0.1, 0.3],   # min values
    #     [160, 0.95, 0.4, 0.9]    # max values
    # )
    # bounds 2
    bounds = (
        [10,  0.1,  0.1, 0.1],   # min values
        [160, 0.95, 0.9, 0.9]    # max values
    )
    optimizer = GlobalBestPSO(
        n_particles=20,
        dimensions=4,
        # option 1
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.5},
        # option 2
        # options = {
        # 'c1': 2.0,      # personal cognitive
        # 'c2': 2.0,      # social/global
        # 'w': 0.9        # initial inertia; will decrease manually
        # },
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(
        pso_objective_function,
        iters=30,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets
    )
    '''Visualize the optimization process'''
    cost_hist = optimizer.cost_history

    # 1) Retrieve full swarm positions history
    pos_hist = optimizer.pos_history  # ⇒ (n_iters, n_particles, dims)

    best_pos_history = []

    # 2) Loop through each iteration
    for positions_i in pos_hist:
        # 2a) Compute cost for all particles at this iteration
        costs_i = pso_objective_function(
            positions_i,
            train_inputs, train_targets,
            val_inputs, val_targets
        )
        # 3) Find best particle index
        best_idx = np.argmin(costs_i)
        # 4) Record its position
        best_pos_history.append(positions_i[best_idx].copy())

    # Convert to array: shape (n_iters, dims)
    best_pos_history = np.vstack(best_pos_history)

    return best_cost, best_pos, cost_hist, best_pos_history

def pso_objective_function(particles, train_inputs, train_targets, val_inputs, val_targets):
    results = []

    for p in particles:
        reservoir_size = int(p[0])
        spectral_radius = p[1]
        sparsity = p[2]
        leak_rate = p[3]

        try:
            esn = ESN(
                input_size=4,
                reservoir_size=reservoir_size,
                output_size=2,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                leak_rate=leak_rate,
                seed=42
            )
            esn.fit(train_inputs, train_targets)
            predictions = esn.predict(val_inputs)

            alpha = 0.7
            mse1 = mean_squared_error(val_targets[:, 0], predictions[:, 0])
            mse2 = mean_squared_error(val_targets[:, 1], predictions[:, 1])
            weighted_mse = alpha * mse2 + (1 - alpha) * mse1

        except Exception as e:
            weighted_mse = 1e6  # large penalty for failed ESN

        results.append(weighted_mse)

    return np.array(results)

def esn_train(data_path, model_name):
    data_for_train = np.load(data_path)
    sample_id = get_sample_ids()
    # Take the example of RH training
    '''Get the train, val, test set of the data'''
    train_set, val_set, test_set =generate_data_sets(sample_id, data_for_train)
    '''Optimize the ESN with PSO'''
    optimal_param = optimize_esn_with_pso(train_set, val_set, model_name)
    # return optimal_param
    return optimal_param
    
def esn_train_and_predict(best_pos, limb_str, train_set, test_set):
    """
    Train ESN using best PSO parameters and predict GRF on test set.

    Parameters:
        best_pos (list or array): Optimized hyperparameters [res_size, sr, sparsity, leak_rate].
        limb_str (str): One of {"RF", "LF", "LH", "RH"}.
        train_set (np.ndarray): Training data.
        test_set (np.ndarray): Test data.

    Returns:
        final_esn (ESN): Trained ESN model.
        grf_predicted (np.ndarray): Predicted GRFs from the test set.
    """
    input_slices = {
        "RF": (16, 20),
        "LF": (20, 24),
        "LH": (24, 28),
        "RH": (28, 32)
    }
    target_slices = {
        "RF": (32, 34),
        "LF": (34, 36),
        "LH": (36, 38),
        "RH": (38, 40)
    }

    if limb_str not in input_slices or limb_str not in target_slices:
        raise ValueError(f"Invalid limb_str '{limb_str}'. Must be one of: {list(input_slices.keys())}")

    # Unpack best parameters
    best_res_size = int(best_pos[0])
    best_sr = best_pos[1]
    best_sp = best_pos[2]
    best_lk = best_pos[3]

    # Get slices
    in_start, in_end = input_slices[limb_str]
    out_start, out_end = target_slices[limb_str]

    # Prepare training and testing data
    train_inputs = train_set[:, in_start:in_end]
    train_targets = train_set[:, out_start:out_end]
    test_inputs = test_set[:, in_start:in_end]
    test_targets = test_set[:, out_start:out_end]

    # Initialize and train ESN
    final_esn = ESN(
        input_size=4,
        reservoir_size=best_res_size,
        output_size=2,
        spectral_radius=best_sr,
        sparsity=best_sp,
        leak_rate=best_lk,
        seed=42
    )
    final_esn.fit(train_inputs, train_targets)

    # Predict
    grf_predicted = final_esn.predict(test_inputs)

    return final_esn, grf_predicted

def save_esn_model(esn_model, save_path):
    """
    Save the ESN model to a file using pickle.

    Parameters:
        esn_model (ESN): The trained ESN model.
        save_path (str): Path to save the model file, e.g., "models/esn_RF.pkl".
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(esn_model, f)


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
    esn_train(data_path,'RH')
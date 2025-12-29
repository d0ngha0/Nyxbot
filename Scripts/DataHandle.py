import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib


def apply_saved_scalers(data, scaler_dir='./Model', slices=[(16, 20), (20, 24), (24, 28), (28, 32)]):
    """
    Apply previously saved scalers to the corresponding column slices of the input data.

    Parameters:
    - data: np.ndarray of shape (n_samples, n_features)
    - scaler_dir: directory where scalers are saved
    - slices: list of (start, end) column indices for each scaler

    Returns:
    - scaled_data: np.ndarray with the same shape as input, scaled
    """
    scaled_data = data.copy()
    for idx, (start, end) in enumerate(slices):
        scaler_path = os.path.join(scaler_dir, f'scaler_{idx}.save')
        scaler = joblib.load(scaler_path)
        scaled_data[:, start:end] = scaler.transform(scaled_data[:, start:end])
    return scaled_data

def compute_segment_means(data):
    """
    Compute the mean of specific segments of a 1D array.

    Parameters:
    - data: np.ndarray of shape (n,)

    Returns:
    - list of means for each specified segment
    """
    means = [
        np.mean(data[0:52]),     # includes index 0 to 51
        np.mean(data[52:82]),    # includes index 52 to 81
        np.mean(data[82:118]),   # includes index 82 to 117
        np.mean(data[118:148]),  # includes index 118 to 147
        np.mean(data[147:])      # includes index 147 to end
    ]
    return np.array(means)

def reshape_to_samples(data, sample_length=70):
    n = len(data)
    num_samples = n // sample_length  # Number of complete samples
    trimmed_data = data[:num_samples * sample_length].copy()
    reshaped = trimmed_data.reshape((num_samples, sample_length))
    return reshaped

def split_by_half_period(data, period, front_half_period, back_half_period):
    '''split the leading and lagging signal in one moving circle'''
    n = data.shape[0] // period
    leading_half = []
    lagging_half = []

    for i in range(n):
        start = i * period
        leading_half.append(data[start : start + front_half_period, :])
        lagging_half.append(data[start + period - back_half_period : start + period, :])

    return np.vstack(leading_half), np.vstack(lagging_half)

def get_sample_ids():
    """
    Generate and shuffle train, validation, and test IDs by combining dataset ranges.

    Returns:
    - tuple: (train_id, val_id, test_id), each is a shuffled np.ndarray
    """
    def generate_split(start, end):
        # Generate the ID list from start to end (inclusive)
        ids = np.arange(start, end + 1)
        
        # Shuffle the ID list randomly
        np.random.seed(42)
        np.random.shuffle(ids)
        
        # Calculate the split sizes
        total_size = len(ids)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
       
        # Split the IDs
        train_set = ids[:train_size]
        val_set = ids[train_size:train_size + val_size]
        test_set = ids[train_size + val_size:]
        
        return train_set, val_set, test_set

    train_ad, val_ad, test_ad = generate_split(0, 50)
    train_lose, val_lose, test_lose = generate_split(51, 171)

    train_id = np.concatenate((train_ad, train_lose))
    val_id = np.concatenate((val_ad, val_lose))
    test_id = np.concatenate((test_ad, test_lose))

    np.random.seed(42)
    np.random.shuffle(train_id)
    np.random.shuffle(val_id)
    np.random.shuffle(test_id)

    return train_id, val_id, test_id

def generate_data_sets(id_tuple, data_for_train, period=140):
    """
    Generate train, validation, and test datasets by slicing data using provided ID indices.

    Parameters:
    - id_tuple: tuple of arrays (train_id, val_id, test_id)
    - data_for_train: np.ndarray, full dataset
    - period: int, length of each segment (default: 140)

    Returns:
    - tuple: (train_set, val_set, test_set), each as a stacked np.ndarray
    """
    train_id, val_id, test_id = id_tuple

    train_set = [data_for_train[period * i: period * (i + 1), :] for i in train_id]
    val_set   = [data_for_train[period * i: period * (i + 1), :] for i in val_id]
    test_set  = [data_for_train[period * i: period * (i + 1), :] for i in test_id]

    return np.vstack(train_set), np.vstack(val_set), np.vstack(test_set)

def normalize_data(data_for_train, save_scaler=None):
    """
    Normalize specific column slices of the input data using MinMaxScaler with feature_range (-1, 1).

    Parameters:
    - data_for_train: np.ndarray, input data to normalize
    - save_scaler: str or None, if provided, scalers will be saved to this directory

    Returns:
    - data_normalized: np.ndarray, data with normalized slices
    """
    # Define the slices to normalize as a constant
    slices = [(16, 20), (20, 24), (24, 28), (28, 32)]

    data_normalized = data_for_train.copy()
    scalers = []

    for idx, (start, end) in enumerate(slices):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        segment = data_for_train[:, start:end]
        data_normalized[:, start:end] = scaler.fit_transform(segment)
        scalers.append(scaler)

        if save_scaler is not None:
            os.makedirs(save_scaler, exist_ok=True)
            joblib.dump(scaler, os.path.join(save_scaler, f'scaler_{idx}.save'))

    return data_normalized

def prepare_datasets(
                     base_path='.',
                     period=140,
                     normal_path='data_for_train.csv',
                     rf_path='RF_limb.npy',
                     lf_path='LF_limb.npy',
                     lh_path='LH_limb.npy',
                     rh_path='RH_limb.npy'):
    """
    Load and preprocess datasets for normal and four abnormal limb conditions.

    Parameters:
        period (int): Number of time steps per sample.
        generate_data_set (function): Function to segment dataset by selected indices.
        normal_path (str): Path to normal data CSV file.
        rf_path (str): Path to RF abnormal data (.npy).
        lf_path (str): Path to LF abnormal data (.npy).
        lh_path (str): Path to LH abnormal data (.npy).
        rh_path (str): Path to RH abnormal data (.npy).

    Returns:
        data_normal, RF_ab, LF_ab, LH_ab, RH_ab (tuple of arrays)
    """

    # Load data
    data_normal = np.loadtxt(os.path.join(base_path, normal_path), delimiter=',')
    RF_ab = np.load(os.path.join(base_path, rf_path))
    LF_ab = np.load(os.path.join(base_path, lf_path))
    LH_ab = np.load(os.path.join(base_path, lh_path))
    RH_ab = np.load(os.path.join(base_path, rh_path))[period*3:, :]  # Crop first 3 periods

    # Shuffle and stack all datasets, preserving their order
    datasets = [data_normal, RF_ab, LF_ab, LH_ab, RH_ab]
    shuffled = [shuffle_data(data) for data in datasets]
    return np.vstack(shuffled)

def shuffle_data(data, period=140, seed=42):
    """
    Shuffle segment indices deterministically and generate a processed dataset from normal data.

    Parameters:
    - data_normal: np.ndarray, original normal data
    - period: int, the segment length used for processing
    - seed: int, random seed for reproducibility (default: 42)

    Returns:
    - np.ndarray, the processed data_normal
    """
    np.random.seed(seed)
    select_id_normal = np.arange(data.shape[0] // period)
    np.random.shuffle(select_id_normal)
    return generate_data_set(select_id_normal, data)

def generate_data_set(train_id, data_for_train, period=140):
    train_set = []
    for i in train_id:
        train_set.append(data_for_train[period * i: period * (i + 1), :])
    return np.vstack(train_set)
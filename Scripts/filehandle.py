
import numpy as np
import os


def get_data_in_directory(path):
    """
    Load and transpose all .npy files directly under `path`,
    printing out each filename and the index it occupies in `data`.
    """
    data = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        # Skip directories and non-.npy files
        if not os.path.isfile(file_path) or not file_name.lower().endswith('.npy'):
            continue

        arr = np.load(file_path).T
        data.append(arr)

        # Print filename and its index in the data list
        print(f"Loaded '{file_name}' into data[{len(data)-1}]")

    return data
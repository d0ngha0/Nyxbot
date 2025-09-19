# LegControl_v4(ESN) - Quadruped Robot Leg Control with Echo State Networks

## Overview

This project implements a hybrid neural network approach for climbing robot adhesive control using Echo State Networks (ESN) and Multi-Layer Perceptrons (MLP). The system focuses on Ground Reaction adhesion (GRA) prediction and adaptive locomotion control for four-legged robots.

## Project Structure

```
LegControl_v4(ESN)/
├── Model/                  # Neural network models
│   ├── cpg.py             # Central Pattern Generator implementation
│   ├── esn.py             # Echo State Network model
│   ├── mlp.py             # Multi-Layer Perceptron model
│   ├── rbf.py             # Radial Basis Function model
│   └── LearnedModel/      # Saved trained models
├── training/              # Training scripts
│   ├── esn_train.py       # ESN training with PSO optimization
│   └── mlp_train.py       # MLP training pipeline
└── Scripts/               # Utility scripts
    ├── DataHandle.py      # Data preprocessing and handling
    ├── DataWashing.py     # Data cleaning utilities
    ├── filehandle.py      # File I/O operations
    └── ForceMapping.py    # Force sensor data mapping
```

## Key Features

### 1. Echo State Network (ESN) Implementation
- **Reservoir Computing**: Implements ESN with configurable reservoir size, spectral radius, sparsity, and leak rate
- **PSO Optimization**: Uses Particle Swarm Optimization to find optimal hyperparameters
- **Multi-limb Support**: Separate models for each limb (RF, LF, LH, RH - Right Front, Left Front, Left Hind, Right Hind)
- **Ground Reaction Adhesion Prediction**: Predicts GRA in Y and Z directions

### 2. Central Pattern Generator (CPG)
- **Biomimetic Control**: Generates rhythmic patterns similar to biological neural circuits
- **Adaptive Parameters**: Time-varying frequency control through `mi_schedule`
- **2D Output**: Generates coordinated oscillations for locomotion control

### 3. Multi-Layer Perceptron (MLP)
- **Prediction Network**: Complements ESN for improved accuracy
- **PyTorch Implementation**: Flexible neural network with configurable layers
- **Model Loading**: Utilities for loading pre-trained models by limb and axis

### 4. Data Processing Pipeline
- **Normalization**: MinMaxScaler with feature range (-1, 1)
- **Segmentation**: Handles periodic data with configurable period length (default: 140 steps)
- **Train/Val/Test Split**: 80/10/10 split with shuffling
- **Missing Data Handling**: Zero-filling algorithms for IMU data

## Installation

### Prerequisites
```bash
pip install numpy
pip install scikit-learn
pip install torch
pip install pyswarms
pip install matplotlib
pip install joblib
```

### Setup
1. Clone the repository
2. Ensure Python 3.7+ is installed
3. Install required dependencies
4. Place training data in `DataForTrain/` directory

## Usage

### Training ESN Models
```python
from training.esn_train import esn_train

# Train ESN for a specific limb
data_path = './DataForTrain/data_for_train.npy'
optimal_params = esn_train(data_path, 'RH')  # Train for Right Hind limb
```

### Using CPG for Pattern Generation
```python
from Model.cpg import generate_cpg_output

# Generate CPG output with time-varying frequency
mi_schedule = [(0, 50, 2.5), (50, 100, 3.0)]  # (start, end, frequency)
cpg_output = generate_cpg_output(140, mi_schedule)
```

### Loading Pre-trained Models
```python
from Model.esn import load_esn_models
from Model.mlp import load_named_models, MLP

# Load ESN models for all limbs
rf_esn, lf_esn, lh_esn, rh_esn = load_esn_models()

# Load MLP models
model_tags = ["RF_Z", "RF_Y", "LF_Z", "LF_Y", "LH_Z", "LH_Y", "RH_Z", "RH_Y"]
mlp_models = load_named_models(MLP, model_tags, "./Model/LearnedModel/")
```

### Data Preprocessing
```python
from Scripts.DataHandle import normalize_data, get_sample_ids, generate_data_sets

# Load and normalize data
data = np.load('data_for_train.npy')
normalized_data = normalize_data(data, save_scaler='./Model/')

# Generate train/val/test splits
sample_ids = get_sample_ids()
train_set, val_set, test_set = generate_data_sets(sample_ids, normalized_data)
```

## Model Architecture

### Echo State Network
- **Input Size**: 4 (joint angles or sensor readings per limb)
- **Reservoir Size**: Optimized via PSO (typically 10-160 neurons)
- **Output Size**: 2 (GRF_Y, GRF_Z)
- **Activation**: Tanh activation function
- **Training**: Ridge regression for output weights

### Multi-Layer Perceptron
- **Input Size**: 70 (half-period of motion data)
- **Hidden Layers**: 2 layers with 32 neurons each
- **Output Size**: 70 (prediction for second half of period)
- **Activation**: ReLU activation

### Central Pattern Generator
- **Dynamics**: 2D oscillator with rotation matrix
- **Output**: Coordinated rhythmic patterns
- **Control**: Frequency modulation through rotation angle

## Data Format

### Input Data Structure
- **Shape**: (N × 140, 40) where N is number of cycles
- **Columns 0-15**: Joint angles for 4 limbs (4 joints each)
- **Columns 16-31**: Joint velocities for 4 limbs
- **Columns 32-39**: Ground Reaction Forces (GRF_Y, GRF_Z for each limb)



## Training Process

1. **Data Preparation**: Load, normalize, and split data
2. **PSO Optimization**: Find optimal ESN hyperparameters
3. **Model Training**: Train ESN with optimal parameters
4. **Validation**: Evaluate on validation set
5. **Model Saving**: Save trained models for deployment

## Performance Metrics

- **MSE (Mean Squared Error)**: Primary optimization metric
- **Weighted MSE**: Custom weighting for Y and Z components
- **Validation Error**: Used for hyperparameter selection

## Applications

- **Adhesive locomotion Control**: Real-time adhesive locomotion control
- **Gait Adaptation**: Adaptive walking patterns
- **Force Prediction**: Ground reaction Adhesion estimation
- **Fault Detection**: Limb malfunction detection through prediction errors


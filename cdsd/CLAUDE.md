# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **CDSD (Causal Discovery with Single-parent Decoding)** for causal representation learning in temporal data. The codebase is from the paper "Causal Representation Learning in Temporal Data via Single-Parent Decoding."

The project consists of two main approaches:
1. **CDSD**: A latent variable model for discovering causal relationships (in `model/` directory)
2. **Varimax-PCMCI**: A baseline comparison method (in `pcmci/` directory)

## Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Key dependencies: PyTorch, tigramite, numpy, scipy, networkx, matplotlib, seaborn, xarray

## Running Experiments

### Training CDSD Model

```bash
# Train with default parameters
python main.py --config-path default_params.json --data-path dataset/data0 --exp-id 0

# Train with custom parameters
python main.py --config-path custom_params.json \
    --data-path dataset/data0 \
    --exp-id 1 \
    --latent \
    --d-z 3 \
    --d-x 10 \
    --tau 2 \
    --batch-size 64 \
    --lr 0.001 \
    --max-iteration 100000
```

Key parameters:
- `--config-path`: JSON file with hyperparameters (overrides command-line args)
- `--data-path`: Path to dataset directory containing `data_x.npy`, `graph.npy`, etc.
- `--exp-path`: Output directory for experiments (default: `causal_climate_exp`)
- `--exp-id`: Experiment ID (creates subdirectory `exp{id}`)
- `--latent`: Use latent variable model (always enabled in current code)
- `--d-z`: Number of latent variables/clusters
- `--d-x`: Number of observed variables per cluster
- `--tau`: Number of past timesteps to consider
- `--instantaneous`: Include instantaneous connections (adds acyclicity constraint)
- `--gpu`: Use GPU for training
- `--no-gt`: Use when ground truth is not available (for real-world data)

### Running Varimax-PCMCI Baseline

```bash
python pcmci/main.py --config-path pcmci/default_params.json \
    --data-path dataset/data0 \
    --exp-id 0
```

### Generating Synthetic Data

```bash
python data_generation/main.py --config-path data_generation_config.json \
    --exp-path dataset/data0 \
    --latent \
    --d-z 3 \
    --d-x 10 \
    --func-type nonlinear \
    --noise-type gaussian
```

Available function types: `linear`, `nonlinear`, `add_nonlinear`, `logistic_map`
Available noise types: `gaussian`, `laplacian`, `uniform`

## Architecture

### Core Model Components

**LatentTSDCD** (`model/tsdcd_latent.py`)
- Main model class implementing the CDSD approach
- Combines: Encoder → Transition Model → Decoder
- Uses Gumbel-Sigmoid sampling for differentiable adjacency matrix learning
- Components:
  - `Autoencoder`: Learns mixing/unmixing functions (X ↔ Z transformation)
  - `TransitionModel`: Learns temporal dynamics between latent variables
  - `Mask`: Learns sparse adjacency matrix via Gumbel-Sigmoid trick

**Key Model Methods:**
- `forward()`: Computes ELBO (reconstruction + KL divergence)
- `get_adj()`: Returns learned adjacency matrix
- `predict()`: Makes predictions for next timestep

### Training Pipeline

**TrainingLatent** (`train_latent.py`)
- Implements training with Augmented Lagrangian Method (ALM/QPM)
- Enforces two constraints:
  1. **Orthogonality constraint**: On mixing matrix W (for identifiability)
  2. **Acyclicity constraint**: On instantaneous connections (if `--instantaneous` is used)
- Training phases:
  1. ALM optimization until constraints converge
  2. Continue training until validation loss stabilizes
  3. Threshold adjacency matrix and fine-tune
- Uses patience-based early stopping

**Key Training Methods:**
- `train_with_QPM()`: Main training loop with constraint optimization
- `train_step()`: Single training iteration with gradient update
- `valid_step()`: Validation evaluation
- `threshold()`: Converts soft adjacency to binary graph

### Data Loading

**DataLoader** (`data_loader.py`)
- Supports two formats: `numpy` (in-memory) and `hdf5` (on-disk sampling)
- Expected numpy files in data directory:
  - `data_x.npy`: Observed data, shape `(n, t, d, d_x)`
  - `data_z.npy`: Latent variables (if available), shape `(n, t, d, d_z)`
  - `graph.npy`: Ground truth causal graph
  - `graph_w.npy`: Ground truth mixing matrix
  - `data_params.json`: Dataset configuration
- Automatically splits data into train/validation based on `ratio_train`, `ratio_valid`

### Constraint Optimization

**ALM Class** (`utils.py`)
- Implements Augmented Lagrangian Method for constrained optimization
- Updates penalty parameter `mu` and Lagrange multipliers `gamma`
- Convergence criteria based on constraint violation decrease

**DAG Constraint** (`dag_optim.py`)
- Uses matrix exponential trick: `trace(exp(A ⊙ A)) - d = 0` for acyclicity
- Applied only when `--instantaneous` flag is used

## Configuration Files

**default_params.json**
- Contains all default hyperparameters
- Override via command-line arguments or custom config files
- Parameters loaded from data generation if `--use-data-config` is set

**Optimizer Notes:**
- Supports: `sgd`, `rmsprop`
- RMSprop is monkey-patched in `prox.py` for gradient projection
- Optimizer is reset when ALM penalty parameter `mu` increases

## Evaluation Metrics

**Causal Graph Metrics** (`metrics.py`):
- `shd()`: Structural Hamming Distance
- `precision_recall()`: Edge precision and recall
- `mcc_latent()`: Mean Correlation Coefficient for latent variable recovery

**Prediction Metrics**:
- MSE: Mean squared error for X_{t+1} prediction
- SMAPE: Symmetric Mean Absolute Percentage Error

## Output Structure

After training, `exp_path/exp{id}/` contains:
- `params.json`: Hyperparameters used
- `results.json`: Final metrics (SHD, precision, recall, MCC, MSE, etc.)
- `train/`: Directory with training plots and checkpoints
- Generated plots showing learned graphs, losses, and predictions

## Important Notes

1. **Latent Model**: The code currently forces `args.latent = True` in main.py:366, so the latent variable model is always used

2. **Data Format**: Observed data has shape `(n, t, d, d_x)` where:
   - `n`: number of samples/trajectories
   - `t`: time steps
   - `d`: number of spatial locations/domains
   - `d_x`: variables per location

3. **Constraints**:
   - W matrix is constrained to be non-negative and orthogonal (columns form orthonormal basis)
   - Adjacency matrix is learned via Gumbel-Sigmoid for differentiability
   - Instantaneous connections require acyclicity constraint (DAG)

4. **Debug Flags**: Several `--debug-gt-*` flags exist for ablation studies using ground truth values (only for synthetic data)

5. **Numerical Precision**: Use `--float` for Float32 instead of Float64 (default is Double)

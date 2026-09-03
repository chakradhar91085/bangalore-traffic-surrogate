# RASTA / Network Lab

RASTA is a reproducible deep-learning prototype for high-fidelity road-network modeling in Indian traffic simulations. It has two layers:

1. A browser dashboard that communicates the traffic state at a glance.
2. A PyTorch graph-temporal neural network that forecasts speed and flow across a Delhi NCR corridor.

The repository uses deterministic synthetic data so the complete pipeline runs without private traffic data or external APIs. It is designed as a strong baseline that can later accept real detector, GPS, weather, or simulation data.

## What the project does

At every simulated time step, each road node has four observed features:

| Feature | Meaning |
| --- | --- |
| `speed` | Estimated vehicle speed at the node in km/h |
| `flow` | Vehicles passing the node during the interval |
| `rain` | Normalized monsoon intensity from 0 to 1 |
| `incident` | Binary disruption indicator |

Given the previous 12 time steps for all nodes, the model predicts speed and flow for the next 3 time steps. In other words, the main input has shape `[batch, 12, 12, 4]` and the output has shape `[batch, 3, 12, 2]`:

```text
12 time steps x 12 road nodes x 4 signals
					|
					v
		spatial graph convolution
					|
					v
		  node-wise temporal GRU
					|
					v
3 future steps x 12 road nodes x 2 targets
```

## Repository layout

```text
.
├── index.html                 Browser dashboard markup
├── styles.css                 Dashboard visual system and responsive layout
├── app.js                     Canvas traffic animation and controls
├── pyproject.toml              Python package metadata and CLI entry points
├── src/rasta_dl/
│   ├── data.py                 Synthetic generator and chronological windows
│   ├── model.py                GraphConv + GRU forecasting model
│   ├── metrics.py              MAE and RMSE calculations
│   ├── train.py                Training, checkpointing, and reports
│   └── evaluate.py             Checkpoint evaluation command
└── tests/test_pipeline.py      Shape, finiteness, and split tests
```

## Quick start

### 1. Start the dashboard

No JavaScript package installation is required:

```powershell
python -m http.server 4173
```

Open `http://localhost:4173` in a browser. The dashboard is a self-contained presentation layer. Its canvas animation is intentionally lightweight and does not make network requests to a model server.

### 2. Install the Python package

Use Python 3.10 or newer. CPU PyTorch is sufficient for this prototype.

```powershell
python -m pip install -e .
```

The package declares `numpy` and `torch` as runtime dependencies. The project was developed and tested with Python 3.13 and PyTorch 2.8 CPU.

### 3. Train a checkpoint

```powershell
python -m rasta_dl.train --epochs 12
```

Training prints the train and validation MAE for every epoch. The best validation checkpoint is saved to:

```text
artifacts/rasta_stgnn.pt
artifacts/rasta_stgnn.json
```

The checkpoint contains the model weights, normalized graph adjacency, and the configuration needed for evaluation. The JSON report contains the complete epoch history and final test metrics.

For a quick smoke run:

```powershell
python -m rasta_dl.train --epochs 2 --output artifacts/smoke.pt
```

### 4. Evaluate a saved model

```powershell
python -m rasta_dl.evaluate artifacts/rasta_stgnn.pt
```

Evaluation reconstructs the same seeded dataset, loads the saved weights, runs only on the held-out chronological test split, and prints MAE and RMSE.

### 5. Run tests

```powershell
python -m pytest -q
```

The tests verify that model output shapes match targets, predictions are finite, and train/validation/test windows remain in chronological order.

## How the data is generated

`make_synthetic_traffic()` creates 12 named nodes representing a simplified Delhi NCR corridor: Karol Bagh, Rajouri Garden, Punjabi Bagh, Dhaula Kuan, Mahipalpur, Aerocity, Vasant Kunj, RK Puram, Lajpat Nagar, ITO, Kashmere Gate, and Connaught Place.

The generator combines:

- Gaussian-shaped morning and evening demand peaks.
- Monsoon rain sampled from a gamma distribution and periodic weather variation.
- Node-specific speed and flow biases.
- Short incident windows that reduce speed and change flow.
- Small sensor noise to prevent a perfectly trivial mapping.

The random seed defaults to `7` and is passed through training, checkpoint metadata, and evaluation. Use a different seed to create another reproducible experiment:

```powershell
python -m rasta_dl.train --seed 42 --epochs 20 --output artifacts/seed42.pt
```

## How the graph model works

`corridor_adjacency()` creates a graph with self-loops, ring-road neighbors, and four corridor shortcuts. The matrix is symmetrically degree-normalized so high-degree nodes do not dominate message passing.

For each historical time step, `GraphConv` aggregates neighboring node features with:

```text
aggregated[t, node] = sum(adjacency[node, neighbor] * features[t, neighbor])
```

A learned linear projection and ReLU turn the aggregated four-channel signal into a 32-dimensional spatial representation. The representations are then rearranged into one temporal sequence per node and passed through a GRU. The final hidden state is projected into the three forecast steps and two targets.

Training uses:

- AdamW optimizer with learning rate `0.002`.
- Smooth L1 loss for robustness to incident-driven outliers.
- Gradient clipping at norm `1.0`.
- Chronological 70/15/15 train, validation, and test splits.
- Best-validation checkpoint selection.

## Dashboard controls

- **Run model** starts or pauses the animated traffic agents.
- **Model** switches the displayed metric profile between Monsoon peak, School release, and Festival surge.
- **Horizon** selects the visible forecast horizon state.
- **Reset** returns the simulation clock and agents to their initial state.
- **Zoom** updates the map scale indicator.
- **Network events** show example disruptions and recovery signals along the corridor.

The dashboard currently uses representative display values rather than loading the PyTorch checkpoint in the browser. A production integration would expose the model through an API or export it to ONNX/WebAssembly, then replace the representative values with live inference results.

## Replacing synthetic data with real data

To connect real traffic data, preserve the `TrafficData` contract in `data.py`:

1. Load a time-indexed array shaped `[time, nodes, features]`.
2. Keep the four feature channels or update `TrafficSTGNN(input_features=...)`.
3. Provide a fixed node ordering matching the adjacency matrix.
4. Split by time before creating windows; never randomly mix future observations into training.
5. Fit feature normalization on the training period only.
6. Store the node list, feature statistics, graph version, and sampling interval with each checkpoint.

For a stronger production model, add missing-value masks, scheduled sampling, multi-head attention, dynamic incident edges, and baselines such as persistence and historical mean. Report MAE and RMSE separately for speed, flow, forecast horizon, and incident periods.

## Limitations

This is a research-grade baseline, not a calibrated traffic management system. The generated signals are not measured observations, the graph is simplified, and the current dashboard is not connected to live inference. Any operational deployment should be validated against held-out Indian traffic data, calibrated by corridor, monitored for drift, and reviewed for safety before decisions are automated.

## License and contribution

This repository is a project prototype. Keep generated files under `artifacts/` out of commits, add tests with new behavior, and include the random seed and evaluation split when reporting model changes.
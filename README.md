# RASTA / Network Lab

A reproducible prototype for accelerating high-fidelity road network modeling in Indian traffic simulations. It combines a live operational dashboard with a graph-temporal deep-learning pipeline that forecasts speed and flow across a Delhi NCR corridor.

The ML pipeline uses a spatial graph convolution over 12 named corridor nodes followed by a node-wise GRU. The included synthetic generator models morning/evening demand, monsoon rain, incidents, node-specific bias, and sensor noise. Chronological splits prevent future leakage, and every training run records a checkpoint plus a JSON report.

## Run locally

From this folder, run:

```powershell
python -m http.server 4173
```

Then open `http://localhost:4173` in a browser. No package installation is required.

## Train the model

Install the Python package in editable mode from this folder:

```powershell
python -m pip install -e .
```

Run a fast CPU training job:

```powershell
python -m rasta_dl.train --epochs 12
python -m rasta_dl.evaluate artifacts/rasta_stgnn.pt
```

The result is written to `artifacts/rasta_stgnn.pt` with metrics in `artifacts/rasta_stgnn.json`. The data is deterministic by seed, so experiments can be reproduced with `--seed`.

Run the model tests with:

```powershell
python -m pytest
```

## Controls

- **Run model** pauses and resumes the traffic agents.
- **Model** changes the metric profile between Monsoon peak, School release, and Festival surge.
- **Reset** returns the simulation clock and agents to their initial state.
- **Zoom** changes the displayed map scale indicator.

# dlproject
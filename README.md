# Accelerating High-Fidelity Road Network Modeling for Indian Traffic Simulations

A deep-learning **surrogate** for Bangalore's road network. Given the scenario on a road (traffic volume, incidents, weather,
roadwork, parking, public transport, signal compliance, pedestrians), it predicts the **congestion class** (Low / Medium / High)
and the **average speed** of all 16 monitored roads in a single forward pass (about 0.2 ms), in place of a slow
traffic simulation run for each scenario. A Streamlit app lets you change a scenario and see the network update.

## Dataset
[Bangalore City Traffic Dataset](https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset) (Kaggle):
8,936 rows, 16 roads in 8 areas, 2022-01-01 to 2024-08-09. The author reports it derives from Government of Karnataka open
traffic data; we have not verified that. The CSV is downloaded automatically with `kagglehub` if `data/raw/` is empty.

## What the EDA found (`notebooks/01_eda.ipynb`)
- No missing values or duplicates; only 58.7% of the date x road grid is observed.
- Congestion Level, Capacity Utilization and Travel Time Index are capped (43.5% / 73.7% / 58.5% of rows at the cap), so they are treated as outcomes, not inputs.
- **No temporal signal** (day-to-day autocorrelation about 0, no weekly pattern) and **no spatial signal** (roads uncorrelated on the same day). A next-day forecaster would have nothing to learn, so this project does not build one.
- Strong same-row signal (traffic volume vs congestion r = 0.84), so the task is a scenario-to-outcome surrogate.
- Weather barely changes congestion (mean 80.3-82.4 across weather types).

## Method
- **Split:** chronological 80/10/10 by date (761 / 95 / 96 days); scalers fit on the training period only.
- **Classes:** Congestion Level < 60 Low, < 90 Medium, otherwise High (fixed cut-offs, because quantile bins collapse on the capped rows).
- **Baselines:** majority/mean, Logistic/Ridge, Random Forest, XGBoost.
- **Deep models** (PyTorch, trained on GPU), each predicting class and speed jointly (cross-entropy + Huber loss):
  - `NodeMLP`: per-road MLP with a learned road embedding.
  - `GraphNet`: the same plus two graph-convolution layers over the road graph (roads linked within an area or within 6 km), using only roads observed that day. One sample is one date's 16-road network snapshot.
- **Regularisation and tuning:** dropout, weight decay, early stopping, a grid over learning rate x width x dropout, and an optimizer comparison (AdamW, Adam, RMSprop, SGD), all selected on validation loss only.

## Results (held-out test set: 900 rows, last 96 dates)
| Model | Accuracy | Macro-F1 | Speed MAE (km/h) | Speed R2 | Inference (ms, CPU) |
|---|---|---|---|---|---|
| Majority / mean | 0.509 | 0.225 | 8.61 | -0.004 | 1.22 |
| Logistic / Ridge | **0.929** | **0.918** | 8.11 | 0.089 | 0.36 |
| Random Forest | 0.920 | 0.907 | **7.69** | 0.165 | 58.61 |
| XGBoost | 0.923 | 0.911 | **7.69** | **0.168** | 2.10 |
| MLP | 0.922 | 0.908 | 7.90 | 0.132 | 0.14 |
| GNN | 0.919 | 0.904 | 7.76 | 0.157 | 0.26 |

Honest reading:
- Every real model reaches about 0.90-0.92 macro-F1. **The simple logistic regression is the best classifier**, and the graph model does not beat the per-road MLP. This matches the EDA: the class is close to a linear function of the road's own conditions and there is no spatial structure to exploit.
- Speed is only weakly predictable (best R2 0.17); treat predicted speed as indicative.
- The deep models' advantage is speed of inference (0.14-0.26 ms for a whole network, versus 59 ms for the Random Forest), not accuracy. GPU inference is slower than CPU at this size (0.33-0.83 ms) because one 16-road snapshot is too small to benefit from it.
- Hyperparameter tuning changed validation loss only within 0.555-0.579; test results with and without tuning are within noise.
- `reports/figures/loss_curves.png` shows overfitting directly: an unregularised 256-unit model drives training loss near 0 while validation loss climbs; the tuned model with dropout and early stopping stays flat.

## Run it
```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -e . jupyter pytest kagglehub
python -m traffic.tune          # grid search + optimizer comparison + overfit reference (~15 min on GPU)
python -m traffic.evaluate      # metrics.json, confusion matrices, latency
python -m traffic.plots         # figures in reports/figures
streamlit run app.py            # web UI
python -m pytest -q             # tests
```
Train a single model: `python -m traffic.train --model gnn --hidden 32 --dropout 0.3`.

## Layout
```
src/traffic/  data.py graph.py baselines.py models.py train.py tune.py evaluate.py plots.py
app.py        Streamlit app (network forecast, what-if scenario, model comparison)
notebooks/    01_eda.ipynb
artifacts/    saved .pth weights, training histories, metrics.json, tuning.csv
reports/figures/  EDA and result figures
tests/        split leakage, scaling, adjacency, model shapes, mask handling
```

## Limitations
One dataset, one city, and data whose provenance we could not verify. The rows look close to independent scenarios rather than a real
time series, so this is a surrogate model, not a traffic forecaster, and the speed predictions are weak. The title's "high-fidelity
simulation" refers to the kind of simulator this surrogate stands in for; we did not run SUMO or any other simulator.

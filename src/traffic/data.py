"""Data ingestion, cleaning, chronological split, scaling and per-date network snapshots."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .graph import build_adjacency

CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "Banglore_traffic_Dataset.csv"
NUMERIC = ["Traffic Volume", "Incident Reports", "Public Transport Usage",
           "Traffic Signal Compliance", "Parking Usage", "Pedestrian and Cyclist Count"]
WEATHER = ["Clear", "Overcast", "Fog", "Rain", "Windy"]
CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_CUTS = (60.0, 90.0)  # Congestion Level < 60 Low, < 90 Medium, else High
SPLITS = (0.8, 0.1, 0.1)


def load_frame(path=CSV):
    if not Path(path).exists():
        import shutil
        import kagglehub
        src = kagglehub.dataset_download("preethamgouda/banglore-city-traffic-dataset")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(src) / "Banglore_traffic_Dataset.csv", path)
    d = pd.read_csv(path, parse_dates=["Date"])
    d = d.rename(columns={"Road/Intersection Name": "Road", "Area Name": "Area"})
    d = d.drop_duplicates().dropna(subset=NUMERIC + ["Average Speed", "Congestion Level"])
    d["Roadwork"] = (d["Roadwork and Construction Activity"] == "Yes").astype(float)
    d["Weather"] = d["Weather Conditions"]
    d["Class"] = np.digitize(d["Congestion Level"], CLASS_CUTS)
    return d.sort_values(["Date", "Road"]).reset_index(drop=True)


@dataclass
class Snapshots:
    """Per-date network snapshots: x [D,N,F], class [D,N], speed [D,N], mask [D,N]."""
    x: np.ndarray
    cls: np.ndarray
    speed: np.ndarray
    mask: np.ndarray
    dates: np.ndarray


@dataclass
class Dataset:
    roads: list
    areas: list
    adj_raw: np.ndarray
    adj: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    speed_mean: float
    speed_std: float
    train: Snapshots
    val: Snapshots
    test: Snapshots
    frame: pd.DataFrame  # cleaned rows with a 'split' column


def node_features(d, roads=None):
    """[n, F] raw features: 6 numeric + roadwork + 5 weather one-hot (road identity is handled by node index)."""
    w = np.stack([(d["Weather"] == k).astype(float).values for k in WEATHER], 1)
    return np.concatenate([d[NUMERIC].values.astype(float), d[["Roadwork"]].values, w], 1)


def build(path=CSV):
    d = load_frame(path)
    roads = sorted(d["Road"].unique())
    areas = [d.loc[d["Road"] == r, "Area"].iloc[0] for r in roads]
    ridx = {r: i for i, r in enumerate(roads)}
    dates = np.sort(d["Date"].unique())
    cut1, cut2 = int(SPLITS[0] * len(dates)), int((SPLITS[0] + SPLITS[1]) * len(dates))
    split_of = {dt: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, dt in enumerate(dates)}
    d["split"] = d["Date"].map(split_of)

    raw = node_features(d)
    tr = (d["split"] == "train").values
    mean, std = raw[tr].mean(0), raw[tr].std(0) + 1e-8
    mean[len(NUMERIC):], std[len(NUMERIC):] = 0.0, 1.0  # only standardise the numeric columns
    feats = (raw - mean) / std
    smean, sstd = d.loc[tr, "Average Speed"].mean(), d.loc[tr, "Average Speed"].std()

    def snaps(names):
        ds = [dt for dt in dates if split_of[dt] in names]
        pos = {dt: i for i, dt in enumerate(ds)}
        n, f = len(roads), feats.shape[1]
        s = Snapshots(np.zeros((len(ds), n, f), np.float32), np.zeros((len(ds), n), np.int64),
                      np.zeros((len(ds), n), np.float32), np.zeros((len(ds), n), bool), np.array(ds))
        sub = d["split"].isin(names).values
        di = d.loc[sub, "Date"].map(pos).values
        ri = d.loc[sub, "Road"].map(ridx).values
        s.x[di, ri] = feats[sub]
        s.cls[di, ri] = d.loc[sub, "Class"].values
        s.speed[di, ri] = ((d.loc[sub, "Average Speed"].values - smean) / sstd)
        s.mask[di, ri] = True
        return s

    adj_raw, adj = build_adjacency(areas)
    return Dataset(roads, areas, adj_raw, adj, mean, std, float(smean), float(sstd),
                   snaps({"train"}), snaps({"val"}), snaps({"test"}), d)


def tabular(ds, split):
    """Flat rows for sklearn baselines: standardised features + road one-hot. Returns X, class, speed (km/h)."""
    d = ds.frame[ds.frame["split"] == split]
    x = (node_features(d) - ds.feat_mean) / ds.feat_std
    road = np.stack([(d["Road"] == r).astype(float).values for r in ds.roads], 1)
    return np.concatenate([x, road], 1), d["Class"].values, d["Average Speed"].values

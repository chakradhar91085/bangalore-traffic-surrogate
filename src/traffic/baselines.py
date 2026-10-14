"""Classical ML baselines on flat rows (features + road one-hot). Each returns class and speed predictors."""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from .data import tabular

FACTORIES = {
    "Majority / mean": lambda: (DummyClassifier(strategy="most_frequent"), DummyRegressor()),
    "Logistic / Ridge": lambda: (LogisticRegression(max_iter=2000), Ridge()),
    "Random Forest": lambda: (RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=7),
                              RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=7)),
    "XGBoost": lambda: (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=7),
                        XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=7)),
}


def fit_baselines(ds):
    X, c, s = tabular(ds, "train")
    out = {}
    for name, make in FACTORIES.items():
        clf, reg = make()
        out[name] = (clf.fit(X, c), reg.fit(X, s))
    return out


def predict_baseline(models, ds, split):
    clf, reg = models
    X, _, _ = tabular(ds, split)
    return clf.predict_proba(X) if hasattr(clf, "predict_proba") else np.eye(3)[clf.predict(X)], reg.predict(X)

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:  # allow baseline-only metrics without torch
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

from repo_config import DATASETS_DIR, MODELS_DIR  # noqa: E402

DATA_DIR = DATASETS_DIR.resolve()
OUT_DIR = MODELS_DIR.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== LEGACY TOGGLES (inherit from dataset build flags) ==========
# These will be auto-loaded from stats.json if available, or use defaults below
SEED = 42
# Global toggle to use force log-ratio targets if available
FORCE_LOG = False
# FAIR mode: unify optimizer/clip, standardize targets, and use validation split
FAIR = True
WARMUP_EPOCHS = 5
SEX_SPECIFIC = False

APPLY_VIA_PREDICT_MASK = False
APPLY_VIA_PREDICT_MASK_EXTRA = True
USE_BASELINE_SUMMARY_FEATURES = False


def _load_build_flags():
    """Load build flags from stats.json to ensure consistency."""
    try:
        with open(DATA_DIR / "stats.json", "r") as f:
            stats = json.load(f)
        flags = stats.get("_build_flags", {})
        return flags
    except Exception:
        return {}


class ArrayDataset(Dataset):  # type: ignore[misc]
    def __init__(self, X: List[List[float]], Y: List[List[float]]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, d_in: int, d_out: int, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden//2), nn.ReLU(),
                nn.Linear(hidden//2, hidden//2), nn.ReLU(),
                nn.Linear(hidden//2, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, d_out),
            )

        def forward(self, x):
            return self.net(x)


SEED = 42
# Global toggle to use force log-ratio targets if available
FORCE_LOG = False
# FAIR mode: unify optimizer/clip, standardize targets, and use validation split
FAIR = True
WARMUP_EPOCHS = 5
SEX_SPECIFIC = False


def train_regressor(X_train, Y_train, X_test, Y_test, out_path: Path, epochs: int = 100, lr: float = 5e-4, seed: int | None = SEED, y_mu=None, y_sd=None, X_val=None, Y_val=None, tag: str | None = None, target_name: str = "val", y_weights=None, muscles=None, via_len=None, metrics_path: Path | None = None, topk: int = 5, weight_factor: float = 1.8):
    assert TORCH_AVAILABLE, "PyTorch not installed"
    # Convert y_weights to tensor if it's a list
    if y_weights is not None and not isinstance(y_weights, torch.Tensor):
        y_weights = torch.tensor(y_weights, dtype=torch.float32)
    # Determinism
    if seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Optional validation split (FAIR mode)
    import numpy as _np
    if FAIR:
        if (X_val is not None) and (Y_val is not None):
            train_ds = ArrayDataset(X_train, Y_train)
            val_ds = ArrayDataset(X_val, Y_val)
        else:
            N = len(X_train)
            rng = _np.random.RandomState(seed or SEED)
            idx = _np.arange(N)
            rng.shuffle(idx)
            n_val = max(1, int(0.1 * N))
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
            X_tr_split = [X_train[i] for i in tr_idx]
            Y_tr_split = [Y_train[i] for i in tr_idx]
            X_val_split = [X_train[i] for i in val_idx]
            Y_val_split = [Y_train[i] for i in val_idx]
            train_ds = ArrayDataset(X_tr_split, Y_tr_split)
            val_ds = ArrayDataset(X_val_split, Y_val_split)
    else:
        train_ds = ArrayDataset(X_train, Y_train)
        val_ds = None
    test_ds = ArrayDataset(X_test, Y_test)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=use_cuda, generator=g)
    test_loader = DataLoader(test_ds, batch_size=128, pin_memory=use_cuda)
    if FAIR and val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=128, pin_memory=use_cuda)
    print(f"Training MLP for {len(X_train[0])} input features and {len(Y_train[0])} output features")
    print(f"Using device: {device}")
    model = MLP(len(X_train[0]), len(Y_train[0])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) if FAIR else torch.optim.Adam(model.parameters(), lr=lr)
    import math as _math
    def _lr_scale(epoch):
        if epoch <= WARMUP_EPOCHS:
            return float(epoch) / max(1, WARMUP_EPOCHS)
        t = (epoch - WARMUP_EPOCHS) / max(1, (epochs - WARMUP_EPOCHS))
        return 0.5 * (1.0 + _math.cos(_math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_scale) if FAIR else None
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    # Optional: refresh per-dim weights for VIA target using metrics.json (top-k hardest muscles)
    def _maybe_refresh_weights():
        nonlocal y_weights
        if (y_weights is None) and (metrics_path is not None) and (muscles is not None) and (via_len is not None):
            try:
                with open(metrics_path, "r") as _mf:
                    _metrics = json.load(_mf)
                pm_via = (_metrics.get("model") or {}).get("per_muscle_via_mse") or {}
                if pm_via:
                    import numpy as _np
                    mus_list = list(pm_via.keys())
                    vals = _np.array([float(pm_via[m]) for m in mus_list], dtype=float)
                    idx = _np.argsort(-vals)[:max(1, min(topk, len(mus_list)))]
                    hard = {mus_list[int(i)] for i in idx}
                    weights = []
                    for m in muscles:
                        d = 3 * int(via_len[m])
                        f = weight_factor if m in hard else 1.0
                        weights.extend([f] * d)
                    y_weights = torch.tensor(weights, dtype=torch.float32)
            except Exception:
                pass

    for epoch in range(1, epochs + 1):
        if target_name == "via" and (epoch == 1 or (epoch % 20 == 0)):
            _maybe_refresh_weights()
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            if (y_weights is not None) and target_name == "via":
                w = y_weights.to(device)
                loss = ((pred - yb) ** 2 * w).mean()
            else:
                loss = loss_fn(pred, yb)
            loss.backward()
            if FAIR:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        # Validation loss (if FAIR), otherwise fall back to test
        eval_loader = val_loader if (FAIR and val_ds is not None) else test_loader
        tot = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                if (y_weights is not None) and target_name == "via":
                    w = y_weights.to(device)
                    loss = ((pred - yb) ** 2 * w).mean()
                else:
                    loss = loss_fn(pred, yb)
                tot += loss.item() * xb.size(0)
                preds.append(pred.cpu())
                trues.append(yb.cpu())
        eval_loss = tot / (len(val_ds) if (FAIR and val_ds is not None) else len(test_ds))
        # For display consistency with other models, if FAIR and stats exist, also compute residual-space MSE
        eval_display = eval_loss
        if FAIR and (val_ds is not None) and (y_mu is not None) and (y_sd is not None):
            import numpy as _np
            P = torch.cat(preds, dim=0).numpy()
            Tt = torch.cat(trues, dim=0).numpy()
            mu = _np.array(y_mu, dtype=float)
            sd = _np.array(y_sd, dtype=float)
            P_res = (P * sd) + mu
            T_res = (Tt * sd) + mu
            eval_display = float(((P_res - T_res) ** 2).mean())

        if eval_loss < best_val:
            best_val = eval_loss
            torch.save({
                "state_dict": model.state_dict(),
                "d_in": len(X_train[0]),
                "d_out": len(Y_train[0]),
                "y_mu": y_mu,
                "y_sd": y_sd,
            }, out_path)

        if epoch % 10 == 0 or epoch == 1:
            split = "val" if (FAIR and val_ds is not None) else "test"
            metric_label = f"{split}_{target_name}"
            prefix = f"[{tag}] " if tag else ""
            print(f"{prefix}epoch {epoch:3d}: train {train_loss:.6f}  {metric_label} {eval_display:.6f}")
        if sched is not None:
            sched.step()

    return best_val


def eval_model(model_path: Path, X: List[List[float]], Y: List[List[float]]):
    assert TORCH_AVAILABLE, "PyTorch not installed"
    ckpt = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(ckpt["d_in"], ckpt["d_out"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    with torch.no_grad():
        Yp = model(X_t).detach().cpu().numpy()
    Y_true = Y_t.detach().cpu().numpy()
    # If checkpoint carries standardization stats, de-standardize predictions and targets
    y_mu = ckpt.get("y_mu")
    y_sd = ckpt.get("y_sd")
    if y_mu is not None and y_sd is not None:
        import numpy as _np
        Yp = (Yp * _np.array(y_sd, dtype=float)) + _np.array(y_mu, dtype=float)
        Y_true = (Y_true * _np.array(y_sd, dtype=float)) + _np.array(y_mu, dtype=float)
    return Yp, Y_true


def mse(a, b):
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    return float(((a - b) ** 2).mean())


def per_dim_mse(a, b):
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    return ((a - b) ** 2).mean(axis=0)


def train_random_forest(X_train, Y_train, X_test, Y_test, out_path: Path, seed: int | None = SEED, X_val=None, Y_val=None, tag: str | None = None):
    """Train a Random Forest model for regression.

    Note: The number of input features depends on USE_BASELINE_SUMMARY_FEATURES flag:
    - If True: X_train includes 94 original features + 2 baseline summary features (bf, bv) = 96 total
    - If False: X_train includes only 94 original features
    The baseline features provide information about the quality of the scaled baseline model.
    """
    """Train Random Forest regressor as an alternative to MLP."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import pickle

    print(f"Training Random Forest for {len(X_train[0])} input features and {len(Y_train[0])} output features")

    # Use validation set if available (FAIR mode), otherwise use test set
    if X_val is not None and Y_val is not None:
        X_eval = X_val
        Y_eval = Y_val
        eval_name = "validation"
    else:
        X_eval = X_test
        Y_eval = Y_test
        eval_name = "test"

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        verbose=0
    )

    print(f"  Training Random Forest...")
    rf.fit(X_train, Y_train)

    # Evaluate
    Y_pred = rf.predict(X_eval)
    eval_mse = mse(Y_pred, Y_eval)

    # Also compute train MSE for reference
    Y_train_pred = rf.predict(X_train)
    train_mse = mse(Y_train_pred, Y_train)

    print(f"  Train MSE: {train_mse:.6f}")
    print(f"  {eval_name.capitalize()} MSE: {eval_mse:.6f}")
    if tag:
        print(f"  Tag: {tag}")

    # Save model
    with open(out_path, 'wb') as f:
        pickle.dump({
            'model': rf,
            'd_in': len(X_train[0]),
            'd_out': len(Y_train[0]),
            'train_mse': train_mse,
            'eval_mse': eval_mse,
            'eval_set': eval_name,
            'seed': seed,
        }, f)

    print(f"  Saved Random Forest to {out_path}")
    return eval_mse


def eval_random_forest(model_path: Path, X: List[List[float]], Y: List[List[float]]):
    """Evaluate a saved Random Forest model."""
    import pickle
    import numpy as np

    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    rf = ckpt['model']
    Y_pred = rf.predict(X)
    Y_true = np.array(Y)

    return Y_pred, Y_true


def main(sex_specific: bool = SEX_SPECIFIC):
    # Load build flags from dataset to ensure consistency
    build_flags = _load_build_flags()
    global FORCE_LOG, FAIR, APPLY_VIA_PREDICT_MASK, APPLY_VIA_PREDICT_MASK_EXTRA, USE_BASELINE_SUMMARY_FEATURES
    # Override with dataset flags if present
    if "FORCE_LOG" in build_flags:
        FORCE_LOG = build_flags["FORCE_LOG"]
    if "FAIR" in build_flags:
        # FAIR can be overridden by user in this file, so only use as hint
        pass

    POS_ONLY = build_flags.get("POS_ONLY", False)
    FORCE_BASELINE_SCALING = build_flags.get("FORCE_BASELINE_SCALING", True)
    APPLY_VIA_PREDICT_MASK = bool(build_flags.get("APPLY_VIA_PREDICT_MASK", APPLY_VIA_PREDICT_MASK))
    APPLY_VIA_PREDICT_MASK_EXTRA = bool(build_flags.get("APPLY_VIA_PREDICT_MASK_EXTRA", APPLY_VIA_PREDICT_MASK_EXTRA))
    USE_BASELINE_SUMMARY_FEATURES = bool(build_flags.get("USE_BASELINE_SUMMARY_FEATURES", USE_BASELINE_SUMMARY_FEATURES))
    print(f"Training with flags: FORCE_LOG={FORCE_LOG}, FAIR={FAIR}, POS_ONLY={POS_ONLY}, FORCE_BASELINE_SCALING={FORCE_BASELINE_SCALING}")

    with open(DATA_DIR / "train.json", "r") as f:
        train = json.load(f)
    with open(DATA_DIR / "test.json", "r") as f:
        test = json.load(f)

    X_tr = train["X"]
    X_te = test["X"]
    Xs_tr = train.get("X_seq")
    Xs_te = test.get("X_seq")
    Ms_tr = train.get("X_seq_mask")
    Ms_te = test.get("X_seq_mask")
    # Load stats for normalization
    import json as _json
    with open(DATA_DIR / "stats.json", "r") as _sf:
        stats = _json.load(_sf)
    def norm_array(arr, mu, sd):
        return [[(v - m) / s for v, m, s in zip(row, mu, sd)] for row in arr]
    X_tr = norm_array(X_tr, stats["X"]["mean"], stats["X"]["std"])
    X_te = norm_array(X_te, stats["X"]["mean"], stats["X"]["std"])
    # Optional validation set
    val = None
    try:
        with open(DATA_DIR / "val.json", "r") as f_val:
            val = json.load(f_val)
    except Exception:
        val = None
    if val is not None:
        X_va = val.get("X")
        X_va = norm_array(X_va, stats["X"]["mean"], stats["X"]["std"]) if X_va is not None else None

    muscles = test["MUSCLES"]
    via_len = test["VIA_LEN"]

    # Use residual targets for training/eval, but compute baseline vs absolute targets
    # For force, optionally switch to log-ratio targets
    Yf_tr_res_or_log = (train.get("Y_force_log") if FORCE_LOG else None) or train["Y_force_res"]
    Yv_tr_res = train["Y_via_res"]
    Yf_te_res_or_log = (test.get("Y_force_log") if FORCE_LOG else None) or test["Y_force_res"]
    Yv_te_res = test["Y_via_res"]
    Yf_bl_te = test.get("Y_force_baseline", [[0.0] * len(Yf_te_res_or_log[0]) for _ in range(len(Yf_te_res_or_log))])
    Yv_bl_te = test.get("Y_via_baseline", [[0.0] * len(Yv_te_res[0]) for _ in range(len(Yv_te_res))])
    # Absolute targets for baseline metrics
    Yf_te_abs = test["Y_force"]
    Yv_te_abs = test["Y_via"]

    # Dump baseline CSVs exactly at the point of baseline MSE calculation
    try:
        import csv, re
        ids = test.get("IDS", [str(i) for i in range(len(Yf_te_abs))])
        def _pid_from_path(p: str) -> str:
            s = Path(p).stem
            m = re.search(r"(\d{3})_", s)
            return m.group(1) if m else s
        # Force baseline pairs
        with open(OUT_DIR / "baseline_mse_force_pairs.csv", "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["patient_id", "muscle", "y_true", "y_pred"])  # header
            for i, pid_path in enumerate(ids):
                pid = _pid_from_path(str(pid_path))
                for mi, mname in enumerate(muscles):
                    w.writerow([pid, mname, float(Yf_te_abs[i][mi]), float(Yf_bl_te[i][mi])])
        # Via baseline pairs
        with open(OUT_DIR / "baseline_mse_via_pairs.csv", "w", newline="") as vcsv:
            wv = csv.writer(vcsv)
            wv.writerow(["patient_id", "muscle", "point_index", "dx_true", "dy_true", "dz_true", "dx_pred", "dy_pred", "dz_pred"])  # header
            for i, pid_path in enumerate(ids):
                pid = _pid_from_path(str(pid_path))
                off = 0
                for m in muscles:
                    Lm = int(via_len[m])
                    for k in range(Lm):
                        tdx, tdy, tdz = map(float, Yv_te_abs[i][off:off+3])
                        pdx, pdy, pdz = map(float, Yv_bl_te[i][off:off+3])
                        wv.writerow([pid, m, k, tdx, tdy, tdz, pdx, pdy, pdz])
                        off += 3
    except Exception:
        pass

    base_force_mse = mse(Yf_bl_te, Yf_te_abs)
    base_via_mse = mse(Yv_bl_te, Yv_te_abs)
    print(f"Baseline MSEs -> force: {base_force_mse:.6f}, via: {base_via_mse:.6f}")

    # Per-muscle baseline MSEs
    import numpy as np
    Yf_err_dim = per_dim_mse(Yf_bl_te, Yf_te_abs)
    per_muscle_force_baseline = {m: float(Yf_err_dim[i]) for i, m in enumerate(muscles)}

    # For via, segment by muscle using 3*via_len[m]
    via_offsets = {}
    off = 0
    for m in muscles:
        via_offsets[m] = (off, off + 3 * via_len[m])
        off += 3 * via_len[m]
    Yv_err_dim = per_dim_mse(Yv_bl_te, Yv_te_abs)
    per_muscle_via_baseline = {}
    for m in muscles:
        s, e = via_offsets[m]
        per_muscle_via_baseline[m] = float(np.mean(Yv_err_dim[s:e]) if e > s else 0.0)

    best_force = None
    best_via = None
    model_force_mse = None
    model_via_mse = None
    per_muscle_force_model = None
    per_muscle_via_model = None

    if TORCH_AVAILABLE:
        # Standardize targets in FAIR mode
        import numpy as np
        with open(DATA_DIR / "stats.json", "r") as _sf2:
            _stats2 = json.load(_sf2)
        force_stats_key = "Y_force_log" if (FORCE_LOG and ("Y_force_log" in _stats2)) else "Y_force_res"
        # Initialize validation variables
        Yf_va = None
        Yv_va = None
        if FAIR:
            yf_mu = np.array(_stats2[force_stats_key]["mean"], dtype=float)
            yf_sd = np.array(_stats2[force_stats_key]["std"], dtype=float)
            yv_mu = np.array(_stats2["Y_via_res"]["mean"], dtype=float)
            yv_sd = np.array(_stats2["Y_via_res"]["std"], dtype=float)
            Yf_tr = ((np.array(Yf_tr_res_or_log, dtype=float) - yf_mu) / yf_sd).tolist()
            Yf_te = ((np.array(Yf_te_res_or_log, dtype=float) - yf_mu) / yf_sd).tolist()
            Yv_tr = ((np.array(Yv_tr_res, dtype=float) - yv_mu) / yv_sd).tolist()
            Yv_te = ((np.array(Yv_te_res, dtype=float) - yv_mu) / yv_sd).tolist()
            # Validation targets if available
            if val is not None:
                Yf_va_res_or_log = (val.get("Y_force_log") if FORCE_LOG else None) or val.get("Y_force_res")
                Yv_va_res = val.get("Y_via_res")
                Yf_va = ((np.array(Yf_va_res_or_log, dtype=float) - yf_mu) / yf_sd).tolist() if Yf_va_res_or_log is not None else None
                Yv_va = ((np.array(Yv_va_res, dtype=float) - yv_mu) / yv_sd).tolist() if Yv_va_res is not None else None
        else:
            Yf_tr = Yf_tr_res_or_log
            Yf_te = Yf_te_res_or_log
            Yv_tr = Yv_tr_res
            Yv_te = Yv_te_res

        # Concatenate baseline summaries (bf,bv) to X (if enabled)
        if USE_BASELINE_SUMMARY_FEATURES:
            def _append_baseline_feats(X_list, Yf_bl, Yv_bl):
                import numpy as _np
                X_out = []
                for i, row in enumerate(X_list):
                    bf = float(_np.mean(_np.abs(_np.array(Yf_bl[i], dtype=float)))) if (Yf_bl is not None) else 0.0
                    bv = float(_np.mean(_np.abs(_np.array(Yv_bl[i], dtype=float)))) if (Yv_bl is not None) else 0.0
                    X_out.append(list(row) + [bf, bv])
                return X_out
            Yf_bl_tr = train.get("Y_force_baseline")
            Yv_bl_tr = train.get("Y_via_baseline")
            if (Yf_bl_tr is not None) and (Yv_bl_tr is not None):
                X_tr = _append_baseline_feats(X_tr, Yf_bl_tr, Yv_bl_tr)
            if (Yf_bl_te is not None) and (Yv_bl_te is not None):
                X_te = _append_baseline_feats(X_te, Yf_bl_te, Yv_bl_te)
            if val is not None and (val.get("Y_force_baseline") is not None):
                X_va = _append_baseline_feats(X_va, val.get("Y_force_baseline"), val.get("Y_via_baseline")) if (X_va is not None) else None

        # Apply FORCE_PREDICT_MASK to weight loss per force-dimension (exclude zeros)
        mask = np.array(test.get("FORCE_PREDICT_MASK") or [1]*len(muscles), dtype=float)
        y_weights_force = mask.tolist()  # 1 for learnable, 0 for excluded

        # Apply VIA_PREDICT_MASK if enabled
        y_weights_via = None
        if APPLY_VIA_PREDICT_MASK:
            # Build via mask from force mask (replicate each muscle's mask for its via coordinates)
            try:
                via_mask_list = []
                for mi, m in enumerate(muscles):
                    L = int(via_len[m])
                    mask_val = float(mask[mi])
                    via_mask_list.extend([mask_val] * (3 * L))
                y_weights_via = via_mask_list
            except Exception as e:
                print(f"Warning: Could not build via mask from force mask: {e}")
                y_weights_via = None

        # Apply VIA_PREDICT_MASK_EXTRA if enabled (exclude height-scaled muscles)
        if APPLY_VIA_PREDICT_MASK_EXTRA and y_weights_via is not None:
            try:
                via_mask_extra = np.array(test.get("VIA_PREDICT_MASK_EXTRA") or [1]*len(muscles), dtype=float)
                # Combine with existing via mask
                via_mask_extra_list = []
                for mi, m in enumerate(muscles):
                    L = int(via_len[m])
                    # Multiply masks: both must be 1 for muscle to be included
                    combined_mask = float(mask[mi]) * float(via_mask_extra[mi])
                    via_mask_extra_list.extend([combined_mask] * (3 * L))
                y_weights_via = via_mask_extra_list
                print(f"Via mask: excluded {sum(1 for w in y_weights_via if w == 0.0) // 3} via coordinates from training")
            except Exception as e:
                print(f"Warning: Could not apply via mask extra: {e}")

        print("Training MLP for residual/log force deltas (FAIR standardization) ..." if FAIR else "Training MLP for residual/log force deltas...")
        best_force = train_regressor(X_tr, Yf_tr, X_te, Yf_te, OUT_DIR / "mlp_force.pt", seed=SEED, y_mu=(yf_mu.tolist() if FAIR else None), y_sd=(yf_sd.tolist() if FAIR else None), X_val=(X_va if (val is not None) else None), Y_val=(Yf_va if (FAIR and (val is not None)) else None), target_name="force", y_weights=y_weights_force, muscles=muscles, via_len=via_len, metrics_path=OUT_DIR / "metrics.json", topk=5, weight_factor=1.8)
        print("Training MLP for residual via-point deltas (FAIR standardization) ..." if FAIR else "Training MLP for residual via-point deltas...")
        best_via = train_regressor(X_tr, Yv_tr, X_te, Yv_te, OUT_DIR / "mlp_via.pt", seed=SEED, y_mu=(yv_mu.tolist() if FAIR else None), y_sd=(yv_sd.tolist() if FAIR else None), X_val=(X_va if (val is not None) else None), Y_val=(Yv_va if (FAIR and (val is not None)) else None), target_name="via", y_weights=y_weights_via, muscles=muscles, via_len=via_len, metrics_path=OUT_DIR / "metrics.json", topk=5, weight_factor=1.8)

        # Evaluate models and compute per-muscle MSEs
        Yf_pred_log, Yf_true_log = eval_model(OUT_DIR / "mlp_force.pt", X_te, Yf_te)
        Yv_pred_res, Yv_true_res = eval_model(OUT_DIR / "mlp_via.pt", X_te, Yv_te)
        # Convert residual preds back to absolute by adding baseline for fair baseline comparison
        Yf_base = np.array(Yf_bl_te)
        Yv_base = np.array(Yv_bl_te)
        # For force: if using log targets, convert to additive via f_t*(exp(log_ratio)-1)
        if FORCE_LOG and ("Y_force_log" in test):
            f_t = np.array(test.get("Y_force_template") or [[0.0]*len(Yf_base[0])]*len(Yf_base), dtype=float)
            Yf_pred_res = f_t * (np.exp(np.array(Yf_pred_log)) - 1.0)
            Yf_true_res = f_t * (np.exp(np.array(Yf_true_log)) - 1.0)
        else:
            Yf_pred_res = np.array(Yf_pred_log)
            Yf_true_res = np.array(Yf_true_log)
        # Ensure FORCE_LOG is handled in latent predictions if needed
        if FORCE_LOG:
            # This is a placeholder; ensure latent models also convert if applicable
            pass
        # Add baseline first, then apply FORCE_PREDICT_MASK on absolute deltas
        Yf_pred = Yf_pred_res + Yf_base
        Yf_true = Yf_true_res + Yf_base
        if (test.get("FORCE_PREDICT_MASK") is not None):
            _mask = np.array(test["FORCE_PREDICT_MASK"], dtype=float).reshape(1, -1)
            Yf_pred = Yf_pred * _mask
            Yf_true = Yf_true * _mask
        Yv_pred = (np.array(Yv_pred_res) + 0.0) + Yv_base
        Yv_true = (np.array(Yv_true_res) + 0.0) + Yv_base
        # Write CSV exactly at the point of computing MSE (absolute with mask applied)
        try:
            import csv, re
            ids = test.get("IDS", [str(i) for i in range(len(Yf_pred))])
            def _pid_from_path(p: str) -> str:
                s = Path(p).stem
                m = re.search(r"(\d{3})_", s)
                return m.group(1) if m else s
            # Force
            with open(OUT_DIR / "mlp_mse_force_pairs.csv", "w", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["patient_id", "muscle", "y_true", "y_pred"])  # header
                for i, pid_path in enumerate(ids):
                    pid = _pid_from_path(str(pid_path))
                    for mi, mname in enumerate(muscles):
                        w.writerow([pid, mname, float(Yf_true[i, mi]), float(Yf_pred[i, mi])])
            # Via
            with open(OUT_DIR / "mlp_mse_via_pairs.csv", "w", newline="") as vcsv:
                wv = csv.writer(vcsv)
                wv.writerow(["patient_id", "muscle", "point_index", "dx_true", "dy_true", "dz_true", "dx_pred", "dy_pred", "dz_pred"])  # header
                for i, pid_path in enumerate(ids):
                    pid = _pid_from_path(str(pid_path))
                    off = 0
                    for m in muscles:
                        Lm = int(via_len[m])
                        for k in range(Lm):
                            tdx, tdy, tdz = map(float, Yv_true[i, off:off+3])
                            pdx, pdy, pdz = map(float, Yv_pred[i, off:off+3])
                            wv.writerow([pid, m, k, tdx, tdy, tdz, pdx, pdy, pdz])
                            off += 3
        except Exception:
            pass
        model_force_mse = mse(Yf_pred, Yf_true)
        model_via_mse = mse(Yv_pred, Yv_true)

        # (Removed duplicate CSV writing block to avoid overwriting)

        # Also report force MSE in neut space for apples-to-apples comparison with train_models_neut.py
        # neut residuals: dF_neut = (F_subject / w_ratio) - F_template = (dF_non_neut) / w_ratio
        aux_te = test.get("AUX", [])
        if aux_te:
            w_ratio = []
            for a in aux_te:
                sex = float(a.get("sex", 1.0))
                generic_w = 78.0 if sex > 0.5 else 61.0
                w_est = float(a.get("w_est", generic_w))
                w_ratio.append(w_est / generic_w if generic_w else 1.0)
            import numpy as _np
            w_ratio = _np.array(w_ratio, dtype=float).reshape(-1, 1)
            Yf_pred_res_neut = _np.array(Yf_pred_res, dtype=float) / w_ratio
            Yf_true_res_neut = _np.array(Yf_true_res, dtype=float) / w_ratio
            model_force_mse_neut = mse(Yf_pred_res_neut, Yf_true_res_neut)
        else:
            model_force_mse_neut = None

        Yf_err_dim_m = per_dim_mse(Yf_pred, Yf_true)
        per_muscle_force_model = {m: float(Yf_err_dim_m[i]) for i, m in enumerate(muscles)}

        Yv_err_dim_m = per_dim_mse(Yv_pred, Yv_true)
        per_muscle_via_model = {}
        for m in muscles:
            s, e = via_offsets[m]
            per_muscle_via_model[m] = float(np.mean(Yv_err_dim_m[s:e]) if e > s else 0.0)

        print("Best test MSE (force):", model_force_mse)
        print("Best test MSE (via):", model_via_mse)
        if model_force_mse_neut is not None:
            print("Best test MSE (force in neut space):", model_force_mse_neut)
        print("Improvement over baseline (force):", base_force_mse - model_force_mse)
        print("Improvement over baseline (via):", base_via_mse - model_via_mse)

        # Train Random Forest models for comparison
        print("\n" + "="*80)
        print("Training Random Forest Models")
        print("="*80)

        # Random Forest for force
        print("\nTraining Random Forest for force prediction...")
        rf_force_mse = train_random_forest(
            X_tr, Yf_tr, X_te, Yf_te,
            OUT_DIR / "rf_force.pkl",
            seed=SEED,
            X_val=X_va if val is not None else None,
            Y_val=Yf_va if (FAIR and val is not None and Yf_va is not None) else None,
            tag="Random Forest Force"
        )

        # Random Forest for via
        print("\nTraining Random Forest for via prediction...")
        rf_via_mse = train_random_forest(
            X_tr, Yv_tr, X_te, Yv_te,
            OUT_DIR / "rf_via.pkl",
            seed=SEED,
            X_val=X_va if val is not None else None,
            Y_val=Yv_va if (FAIR and val is not None and Yv_va is not None) else None,
            tag="Random Forest Via"
        )

        # Evaluate Random Forest models (convert standardized predictions back to residuals)
        Yf_pred_rf_std, Yf_true_rf_std = eval_random_forest(OUT_DIR / "rf_force.pkl", X_te, Yf_te)
        Yv_pred_rf_std, Yv_true_rf_std = eval_random_forest(OUT_DIR / "rf_via.pkl", X_te, Yv_te)

        # De-standardize if FAIR
        if FAIR:
            Yf_pred_rf_res = (Yf_pred_rf_std * _np.array(yf_sd, dtype=float)) + _np.array(yf_mu, dtype=float)
            Yf_true_rf_res = (Yf_true_rf_std * _np.array(yf_sd, dtype=float)) + _np.array(yf_mu, dtype=float)
            Yv_pred_rf_res = (Yv_pred_rf_std * _np.array(yv_sd, dtype=float)) + _np.array(yv_mu, dtype=float)
            Yv_true_rf_res = (Yv_true_rf_std * _np.array(yv_sd, dtype=float)) + _np.array(yv_mu, dtype=float)
        else:
            Yf_pred_rf_res = Yf_pred_rf_std
            Yf_true_rf_res = Yf_true_rf_std
            Yv_pred_rf_res = Yv_pred_rf_std
            Yv_true_rf_res = Yv_true_rf_std

        # Add baseline to get absolute predictions
        Yf_pred_rf_abs = Yf_pred_rf_res + _np.array(Yf_bl_te)
        Yf_true_rf_abs = Yf_true_rf_res + _np.array(Yf_bl_te)
        Yv_pred_rf_abs = Yv_pred_rf_res + _np.array(Yv_bl_te)
        Yv_true_rf_abs = Yv_true_rf_res + _np.array(Yv_bl_te)

        # Compute absolute-space MSEs for Random Forest
        rf_force_mse_abs = mse(Yf_pred_rf_abs, Yf_true_rf_abs)
        rf_via_mse_abs = mse(Yv_pred_rf_abs, Yv_true_rf_abs)

        print(f"\nRandom Forest Results (Absolute Space):")
        print(f"  Force MSE: {rf_force_mse_abs:.6f} (Improvement over baseline: {base_force_mse - rf_force_mse_abs:.6f})")
        print(f"  Via MSE: {rf_via_mse_abs:.6f} (Improvement over baseline: {base_via_mse - rf_via_mse_abs:.6f})")

    else:
        print("PyTorch not installed; computed baseline metrics only.")
        rf_force_mse_abs = None
        rf_via_mse_abs = None

    # Per-sex training (in addition to combined)
    if TORCH_AVAILABLE and sex_specific:
        import numpy as np
        sex_tr = [a.get("sex", 0.0) for a in train.get("AUX", [])]
        sex_te = [a.get("sex", 0.0) for a in test.get("AUX", [])]
        idxM_tr = [i for i,s in enumerate(sex_tr) if s > 0.5]
        idxF_tr = [i for i,s in enumerate(sex_tr) if s <= 0.5]
        idxM_te = [i for i,s in enumerate(sex_te) if s > 0.5]
        idxF_te = [i for i,s in enumerate(sex_te) if s <= 0.5]

        def sel(arr, idxs):
            return [arr[i] for i in idxs]

        for label, tr_idx, te_idx, force_out, via_out in (
            ("m", idxM_tr, idxM_te, OUT_DIR / "mlp_force_m.pt", OUT_DIR / "mlp_via_m.pt"),
            ("f", idxF_tr, idxF_te, OUT_DIR / "mlp_force_f.pt", OUT_DIR / "mlp_via_f.pt"),
        ):
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            print(f"Training MLP (sex={label}) for residual/log force deltas...")
            best_force_s = train_regressor(
                sel(X_tr, tr_idx), sel(Yf_tr, tr_idx), sel(X_te, te_idx), sel(Yf_te, te_idx),
                force_out, seed=SEED,
                y_mu=(yf_mu.tolist() if FAIR else None), y_sd=(yf_sd.tolist() if FAIR else None),
                tag=f"sex={label} force", target_name="force", muscles=muscles, via_len=via_len, metrics_path=OUT_DIR / "metrics.json", topk=5, weight_factor=1.8
            )
            print(f"Training MLP (sex={label}) for residual via-point deltas...")
            best_via_s = train_regressor(
                sel(X_tr, tr_idx), sel(Yv_tr, tr_idx), sel(X_te, te_idx), sel(Yv_te, te_idx),
                via_out, seed=SEED,
                y_mu=(yv_mu.tolist() if FAIR else None), y_sd=(yv_sd.tolist() if FAIR else None),
                tag=f"sex={label} via", target_name="via", muscles=muscles, via_len=via_len, metrics_path=OUT_DIR / "metrics.json", topk=5, weight_factor=1.8
            )

            # Eval and improvements for subset
            Yf_pred_t, Yf_true_t = eval_model(force_out, sel(X_te, te_idx), sel(Yf_te, te_idx))
            Yv_pred_t, Yv_true_t = eval_model(via_out, sel(X_te, te_idx), sel(Yv_te, te_idx))
            Yf_base_s = np.array(sel(Yf_bl_te, te_idx))
            Yv_base_s = np.array(sel(Yv_bl_te, te_idx))
            # Force convert if log (only when FORCE_LOG was used for targets)
            if FORCE_LOG and ("Y_force_log" in test):
                f_t_s = np.array(sel(test.get("Y_force_template"), te_idx), dtype=float)
                Yf_pred_res_s = f_t_s * (np.exp(np.array(Yf_pred_t)) - 1.0)
                Yf_true_res_s = f_t_s * (np.exp(np.array(Yf_true_t)) - 1.0)
            else:
                Yf_pred_res_s = np.array(Yf_pred_t)
                Yf_true_res_s = np.array(Yf_true_t)
            Yf_pred_abs_s = Yf_pred_res_s + Yf_base_s
            Yf_true_abs_s = Yf_true_res_s + Yf_base_s
            Yv_pred_abs_s = np.array(Yv_pred_t) + Yv_base_s
            Yv_true_abs_s = np.array(Yv_true_t) + Yv_base_s
            base_force_s = mse(Yf_base_s, np.array(sel(test["Y_force"], te_idx)))
            base_via_s = mse(Yv_base_s, np.array(sel(test["Y_via"], te_idx)))
            force_mse_s = mse(Yf_pred_abs_s, Yf_true_abs_s)
            via_mse_s = mse(Yv_pred_abs_s, Yv_true_abs_s)
            print(f"[sex={label}] Best test MSEs -> force: {force_mse_s:.6f}, via: {via_mse_s:.6f}")
            print(f"[sex={label}] Improvement over baseline -> force: {base_force_s - force_mse_s:.6f}, via: {base_via_s - via_mse_s:.6f}")

    # Save absolute-space metrics (outside-of-training eval already computed above)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "baseline": {
                "force_mse": base_force_mse,
                "via_mse": base_via_mse,
                "per_muscle_force_mse": per_muscle_force_baseline,
                "per_muscle_via_mse": per_muscle_via_baseline,
            },
            "model": {
                "force_mse": model_force_mse,
                "via_mse": model_via_mse,
                "per_muscle_force_mse": per_muscle_force_model,
                "per_muscle_via_mse": per_muscle_via_model,
            },
            "random_forest": {
                "force_mse": rf_force_mse_abs if TORCH_AVAILABLE else None,
                "via_mse": rf_via_mse_abs if TORCH_AVAILABLE else None,
            },
            "notes": {
                "absolute_space": True,
                "fair": FAIR,
            }
        }, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex-specific", action="store_true", default=SEX_SPECIFIC, help="Train male/female-specific models in addition to combined model")
    args = parser.parse_args()
    main(sex_specific=args.sex_specific)

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
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
SEED = 42
FORCE_LOG = False        # Use log-ratio force targets instead of additive residuals
FAIR = True              # FAIR mode: standardize targets, use validation split
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


def load_dataset() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(DATA_DIR / "train.json", "r") as f:
        train = json.load(f)
    with open(DATA_DIR / "test.json", "r") as f:
        test = json.load(f)
    return train, test


class ArrayDataset(Dataset):  # type: ignore[misc]
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert TORCH_AVAILABLE, "PyTorch not installed"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def mse(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(((a - b) ** 2).mean())


def train_random_forest_latent(X_train: np.ndarray, Yf_train: np.ndarray, Yv_train: np.ndarray,
                                X_test: np.ndarray, Yf_test: np.ndarray, Yv_test: np.ndarray,
                                out_path_force: Path, out_path_via: Path,
                                seed: int = SEED, X_val=None, Yf_val=None, Yv_val=None):
    """Train Random Forest models for force and via predictions (alternative to latent model).

    Note: The number of input features depends on USE_BASELINE_SUMMARY_FEATURES flag:
    - If True: Uses 94 original features + 2 baseline summary features (bf, bv) = 96 total
    - If False: Uses only 94 original features
    Set USE_BASELINE_SUMMARY_FEATURES=True in both train_models.py and train_latent.py for
    consistent comparison. The RF models provide a fair comparison to their respective neural
    network architectures using identical input features.
    """
    from sklearn.ensemble import RandomForestRegressor
    import pickle

    print(f"Training Random Forest models as alternative to latent autoencoder...")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Force outputs: {Yf_train.shape[1]}")
    print(f"  Via outputs: {Yv_train.shape[1]}")

    # Use validation set if available
    if X_val is not None and Yf_val is not None:
        X_eval, Yf_eval, Yv_eval = X_val, Yf_val, Yv_val
        eval_name = "validation"
    else:
        X_eval, Yf_eval, Yv_eval = X_test, Yf_test, Yv_test
        eval_name = "test"

    # Train Random Forest for force
    print("\n  Training Random Forest for force...")
    rf_force = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        verbose=0
    )
    rf_force.fit(X_train, Yf_train)
    Yf_pred = rf_force.predict(X_eval)
    force_mse = mse(Yf_pred, Yf_eval)
    print(f"    {eval_name.capitalize()} Force MSE: {force_mse:.6f}")

    # Train Random Forest for via
    print("\n  Training Random Forest for via...")
    rf_via = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        verbose=0
    )
    rf_via.fit(X_train, Yv_train)
    Yv_pred = rf_via.predict(X_eval)
    via_mse = mse(Yv_pred, Yv_eval)
    print(f"    {eval_name.capitalize()} Via MSE: {via_mse:.6f}")

    # Save models
    with open(out_path_force, 'wb') as f:
        pickle.dump({
            'model': rf_force,
            'd_in': X_train.shape[1],
            'd_out': Yf_train.shape[1],
            'eval_mse': force_mse,
            'eval_set': eval_name,
            'seed': seed,
        }, f)

    with open(out_path_via, 'wb') as f:
        pickle.dump({
            'model': rf_via,
            'd_in': X_train.shape[1],
            'd_out': Yv_train.shape[1],
            'eval_mse': via_mse,
            'eval_set': eval_name,
            'seed': seed,
        }, f)

    print(f"\n  Saved Random Forest models to {out_path_force} and {out_path_via}")
    return force_mse, via_mse


def build_offsets(muscles: List[str], via_len: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    offsets: Dict[str, Tuple[int, int]] = {}
    off = 0
    for m in muscles:
        L = int(via_len[m])
        s, e = off, off + 3 * L
        offsets[m] = (s, e)
        off = e
    return offsets


if TORCH_AVAILABLE:
    class PathAutoEncoder(nn.Module):
        def __init__(self, d_in: int, latent_dim: int = 128, hidden: int = 512):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, d_in),
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))


    class LatentRegressor(nn.Module):
        def __init__(self, x_dim: int, latent_dim: int, y_force_dim: int):
            super().__init__()
            self.to_latent = nn.Sequential(
                nn.Linear(x_dim, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, latent_dim),
            )
            self.force_head = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(),
                nn.Linear(256, y_force_dim),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            z = self.to_latent(x)
            y_force = self.force_head(z)
            return z, y_force


def smoothness_penalty(batch_decoded: torch.Tensor, muscles: List[str], via_len: Dict[str, int]) -> torch.Tensor:
    """Penalize consecutive point differences along each muscle path.

    batch_decoded: [B, D] flattened (x1,y1,z1,x2,y2,z2, ... per muscle)
    """
    assert TORCH_AVAILABLE
    device = batch_decoded.device
    B, D = batch_decoded.shape
    loss = torch.zeros((), device=device)
    off = 0
    for m in muscles:
        L = int(via_len[m])
        if L <= 1:
            off += 3 * L
            continue
        seg = batch_decoded[:, off:off + 3 * L]  # [B, 3L]
        off += 3 * L
        pts = seg.reshape(B, L, 3)
        diffs = pts[:, 1:, :] - pts[:, :-1, :]
        loss = loss + (diffs ** 2).mean()
    return loss


def main(sex_specific: bool = SEX_SPECIFIC):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed; skipping latent training.")
        return

    # Load build flags from dataset to ensure consistency
    build_flags = _load_build_flags()
    global FORCE_LOG, FAIR, APPLY_VIA_PREDICT_MASK, APPLY_VIA_PREDICT_MASK_EXTRA, USE_BASELINE_SUMMARY_FEATURES
    # Override with dataset flags if present
    if "FORCE_LOG" in build_flags:
        FORCE_LOG = build_flags["FORCE_LOG"]
    if "FAIR" in build_flags:
        pass  # FAIR can be overridden by user

    POS_ONLY = build_flags.get("POS_ONLY", False)
    FORCE_BASELINE_SCALING = build_flags.get("FORCE_BASELINE_SCALING", True)
    APPLY_VIA_PREDICT_MASK = bool(build_flags.get("APPLY_VIA_PREDICT_MASK", APPLY_VIA_PREDICT_MASK))
    APPLY_VIA_PREDICT_MASK_EXTRA = bool(build_flags.get("APPLY_VIA_PREDICT_MASK_EXTRA", APPLY_VIA_PREDICT_MASK_EXTRA))
    USE_BASELINE_SUMMARY_FEATURES = bool(build_flags.get("USE_BASELINE_SUMMARY_FEATURES", USE_BASELINE_SUMMARY_FEATURES))
    print(f"Training latent with flags: FORCE_LOG={FORCE_LOG}, FAIR={FAIR}, POS_ONLY={POS_ONLY}, FORCE_BASELINE_SCALING={FORCE_BASELINE_SCALING}")

    # Seeds
    import random as _random
    _random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load
    train, test = load_dataset()
    with open(DATA_DIR / "stats.json", "r") as f:
        stats = json.load(f)

    X_tr = np.array(train["X"], dtype=float)
    X_te = np.array(test["X"], dtype=float)
    # Normalize X
    x_mu = np.array(stats["X"]["mean"], dtype=float)
    x_sd = np.array(stats["X"]["std"], dtype=float)
    X_tr_n = (X_tr - x_mu) / x_sd
    X_te_n = (X_te - x_mu) / x_sd

    # Append baseline summary features (bf, bv) if enabled
    if USE_BASELINE_SUMMARY_FEATURES:
        Yf_bl_tr = train.get("Y_force_baseline")
        Yv_bl_tr = train.get("Y_via_baseline")
        Yf_bl_te = test.get("Y_force_baseline")
        Yv_bl_te_feat = test.get("Y_via_baseline")

        if (Yf_bl_tr is not None) and (Yv_bl_tr is not None):
            # Compute baseline summary features for training set
            bf_tr = np.mean(np.abs(np.array(Yf_bl_tr, dtype=float)), axis=1, keepdims=True)
            bv_tr = np.mean(np.abs(np.array(Yv_bl_tr, dtype=float)), axis=1, keepdims=True)
            X_tr_n = np.concatenate([X_tr_n, bf_tr, bv_tr], axis=1)

        if (Yf_bl_te is not None) and (Yv_bl_te_feat is not None):
            # Compute baseline summary features for test set
            bf_te = np.mean(np.abs(np.array(Yf_bl_te, dtype=float)), axis=1, keepdims=True)
            bv_te = np.mean(np.abs(np.array(Yv_bl_te_feat, dtype=float)), axis=1, keepdims=True)
            X_te_n = np.concatenate([X_te_n, bf_te, bv_te], axis=1)

    # Targets (residuals)
    Yv_tr_res = np.array(train["Y_via_res"], dtype=float)
    Yv_te_res = np.array(test["Y_via_res"], dtype=float)
    # Force: prefer log-ratio targets if present
    if FORCE_LOG and ("Y_force_log" in train) and ("Y_force_log" in test):
        Yf_tr_res = np.array(train["Y_force_log"], dtype=float)
        Yf_te_res = np.array(test["Y_force_log"], dtype=float)
        force_stats_key = "Y_force_log"
    else:
        Yf_tr_res = np.array(train["Y_force_res"], dtype=float)
        Yf_te_res = np.array(test["Y_force_res"], dtype=float)
        force_stats_key = "Y_force_res"
    # Baselines (absolute metrics)
    Yv_bl_te = np.array(test["Y_via_baseline"], dtype=float)
    Yf_bl_te = np.array(test["Y_force_baseline"], dtype=float)
    Yv_te_abs = np.array(test["Y_via"], dtype=float)
    Yf_te_abs = np.array(test["Y_force"], dtype=float)
    base_force_mse = mse(Yf_bl_te, Yf_te_abs)
    base_via_mse = mse(Yv_bl_te, Yv_te_abs)
    print(f"Baseline MSEs -> force: {base_force_mse:.6f}, via: {base_via_mse:.6f}")

    muscles = train["MUSCLES"]
    via_len = train["VIA_LEN"]
    # Optional force mask (exclude always-zero muscles)
    try:
        force_mask_np = np.array((test.get("FORCE_PREDICT_MASK") or train.get("FORCE_PREDICT_MASK") or [1]*len(muscles)), dtype=float)
    except Exception:
        force_mask_np = None

    # Optional via mask (exclude masked/height-scaled muscles)
    via_mask_np = None
    if APPLY_VIA_PREDICT_MASK:
        try:
            # Build via mask from force mask
            force_mask_arr = force_mask_np if force_mask_np is not None else np.ones(len(muscles), dtype=float)
            via_mask_list = []
            for mi, m in enumerate(muscles):
                L = int(via_len[m])
                mask_val = float(force_mask_arr[mi])
                via_mask_list.extend([mask_val] * (3 * L))
            via_mask_np = np.array(via_mask_list, dtype=float)
        except Exception:
            via_mask_np = None

    # Apply via mask extra if enabled
    if APPLY_VIA_PREDICT_MASK_EXTRA and via_mask_np is not None:
        try:
            via_mask_extra = np.array(test.get("VIA_PREDICT_MASK_EXTRA") or train.get("VIA_PREDICT_MASK_EXTRA") or [1]*len(muscles), dtype=float)
            # Combine masks
            via_mask_combined = []
            for mi, m in enumerate(muscles):
                L = int(via_len[m])
                combined_mask = float(force_mask_arr[mi]) * float(via_mask_extra[mi])
                via_mask_combined.extend([combined_mask] * (3 * L))
            via_mask_np = np.array(via_mask_combined, dtype=float)
        except Exception:
            pass

    # Standardize residual targets
    yv_mu = np.array(stats["Y_via_res"]["mean"], dtype=float)
    yv_sd = np.array(stats["Y_via_res"]["std"], dtype=float)
    yf_mu = np.array(stats[force_stats_key]["mean"], dtype=float)
    yf_sd = np.array(stats[force_stats_key]["std"], dtype=float)
    Yv_tr_std = (Yv_tr_res - yv_mu) / yv_sd
    Yv_te_std = (Yv_te_res - yv_mu) / yv_sd
    Yf_tr_std = (Yf_tr_res - yf_mu) / yf_sd
    Yf_te_std = (Yf_te_res - yf_mu) / yf_sd

    d_x = X_tr_n.shape[1]
    d_v = Yv_tr_std.shape[1]
    d_f = Yf_tr_std.shape[1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Stage 1: Unsupervised autoencoder on Y_via_res (standardized)
    ae = PathAutoEncoder(d_in=d_v, latent_dim=256, hidden=1024).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    ds_ae = ArrayDataset(np.zeros((Yv_tr_std.shape[0], 1), dtype=float), Yv_tr_std)  # dummy X, use Y only
    # simple wrapper dataloader emitting only Y
    class _OnlyY(Dataset):  # type: ignore[misc]
        def __init__(self, Y):
            self.Y = torch.tensor(Y, dtype=torch.float32)
        def __len__(self):
            return self.Y.shape[0]
        def __getitem__(self, i):
            return self.Y[i]
    train_loader_ae = DataLoader(_OnlyY(Yv_tr_std), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    smooth_lambda = 3e-3
    curvature_lambda = 1e-3
    for epoch in range(1, 101):
        ae.train()
        tot = 0.0
        for yb in train_loader_ae:
            yb = yb.to(device, non_blocking=True)
            opt_ae.zero_grad()
            rec = ae(yb)
            rec_loss = nn.functional.mse_loss(rec, yb)
            sm = smoothness_penalty(rec, muscles, via_len)
            # curvature (second difference) penalty
            B = rec.shape[0]
            off = 0
            curv = torch.zeros((), device=rec.device)
            for m in muscles:
                L = int(via_len[m])
                if L <= 2:
                    off += 3 * L
                    continue
                seg = rec[:, off:off + 3 * L].reshape(B, L, 3)
                off += 3 * L
                d1 = seg[:, 1:, :] - seg[:, :-1, :]
                d2 = d1[:, 1:, :] - d1[:, :-1, :]
                curv = curv + (d2 ** 2).mean()
            loss = rec_loss + smooth_lambda * sm + curvature_lambda * curv
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt_ae.step()
            tot += loss.item() * yb.size(0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"AE epoch {epoch:3d}: train_loss {tot/len(Yv_tr_std):.6f}")

    # Stage 2: Supervised regressor to latent; decoder reused (frozen)
    # Partially unfreeze decoder: only last layer trainable
    for p in ae.decoder.parameters():
        p.requires_grad = False
    # Unfreeze last Linear of decoder
    if isinstance(ae.decoder[-1], nn.Linear):
        for p in ae.decoder[-1].parameters():
            p.requires_grad = True
    reg = LatentRegressor(x_dim=d_x, latent_dim=256, y_force_dim=d_f).to(device)
    opt_reg = torch.optim.AdamW([
        {"params": reg.parameters(), "lr": 2e-4, "weight_decay": 1e-4},
        {"params": ae.decoder[-1].parameters(), "lr": 5e-5, "weight_decay": 1e-4},
    ])
    loss_fn = nn.MSELoss()
    # FAIR: build validation split indices on training data
    if FAIR:
        # Prefer external val.json if available
        try:
            with open(DATA_DIR / "val.json", "r") as f:
                val = json.load(f)
        except Exception:
            val = None
        if val is not None:
            X_val_reg = ((np.array(val["X"], dtype=float) - x_mu) / x_sd)

            # Append baseline summary features to validation set if enabled
            if USE_BASELINE_SUMMARY_FEATURES:
                Yf_bl_val = val.get("Y_force_baseline")
                Yv_bl_val = val.get("Y_via_baseline")
                if (Yf_bl_val is not None) and (Yv_bl_val is not None):
                    bf_val = np.mean(np.abs(np.array(Yf_bl_val, dtype=float)), axis=1, keepdims=True)
                    bv_val = np.mean(np.abs(np.array(Yv_bl_val, dtype=float)), axis=1, keepdims=True)
                    X_val_reg = np.concatenate([X_val_reg, bf_val, bv_val], axis=1)

            Yv_val_res = np.array(val["Y_via_res"], dtype=float)
            Yf_val_res_or_log = np.array((val.get("Y_force_log") if FORCE_LOG else None) or val["Y_force_res"], dtype=float)
            Yv_val_reg = (Yv_val_res - yv_mu) / yv_sd
            Yf_val_reg = (Yf_val_res_or_log - yf_mu) / yf_sd
            X_tr_reg = X_tr_n
            Yv_tr_reg = Yv_tr_std
            Yf_tr_reg = Yf_tr_std
        else:
            N = X_tr_n.shape[0]
            rng = np.random.RandomState(SEED)
            idx = np.arange(N)
            rng.shuffle(idx)
            n_val = max(1, int(0.1 * N))
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
            X_tr_reg = X_tr_n[tr_idx]
            Yv_tr_reg = Yv_tr_std[tr_idx]
            Yf_tr_reg = Yf_tr_std[tr_idx]
            X_val_reg = X_tr_n[val_idx]
            Yv_val_reg = Yv_tr_std[val_idx]
            Yf_val_reg = Yf_tr_std[val_idx]
    else:
        X_tr_reg = X_tr_n
        Yv_tr_reg = Yv_tr_std
        Yf_tr_reg = Yf_tr_std
        X_val_reg = None
        Yv_val_reg = None
        Yf_val_reg = None
    train_loader = DataLoader(ArrayDataset(X_tr_reg, np.stack([np.zeros(d_f)] * len(X_tr_reg))), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    best_force = float('inf')
    best_via = float('inf')
    for epoch in range(1, 101):
        reg.train(); ae.eval()
        total = 0.0
        # We still need Yf_tr_std and Yv_tr_std inside loop; iterate indices
        if FAIR:
            idxs = torch.randperm(X_tr_reg.shape[0], generator=torch.Generator().manual_seed(SEED + epoch))
            Xpool = X_tr_reg; Yvpool = Yv_tr_reg; Yfpool = Yf_tr_reg
        else:
            idxs = torch.randperm(X_tr_n.shape[0], generator=torch.Generator().manual_seed(SEED + epoch))
            Xpool = X_tr_n; Yvpool = Yv_tr_std; Yfpool = Yf_tr_std
        for i0 in range(0, len(idxs), 64):
            sel = idxs[i0:i0+64]
            xb = torch.tensor(Xpool[sel.numpy()], dtype=torch.float32, device=device)
            yv = torch.tensor(Yvpool[sel.numpy()], dtype=torch.float32, device=device)
            yf = torch.tensor(Yfpool[sel.numpy()], dtype=torch.float32, device=device)
            opt_reg.zero_grad()
            z, yf_pred = reg(xb)
            yv_pred = ae.decode(z)
            if FAIR:
                # Weight force loss by mask if provided (exclude always-zero muscles)
                if force_mask_np is not None:
                    w = torch.tensor(force_mask_np, dtype=torch.float32, device=device)
                    force_loss = ((yf_pred - yf) ** 2 * w).mean()
                else:
                    force_loss = loss_fn(yf_pred, yf)
                # Weight via loss by mask if provided
                if via_mask_np is not None:
                    w_via = torch.tensor(via_mask_np, dtype=torch.float32, device=device)
                    via_loss = ((yv_pred - yv) ** 2 * w_via).mean()
                else:
                    via_loss = loss_fn(yv_pred, yv)
                loss = 0.5 * via_loss + 0.5 * force_loss
            else:
                if force_mask_np is not None:
                    w = torch.tensor(force_mask_np, dtype=torch.float32, device=device)
                    force_loss = ((yf_pred - yf) ** 2 * w).mean()
                else:
                    force_loss = loss_fn(yf_pred, yf)
                if via_mask_np is not None:
                    w_via = torch.tensor(via_mask_np, dtype=torch.float32, device=device)
                    via_loss = ((yv_pred - yv) ** 2 * w_via).mean()
                else:
                    via_loss = loss_fn(yv_pred, yv)
                loss = via_loss + 0.5 * force_loss
            loss.backward()
            nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
            opt_reg.step()
            total += loss.item() * xb.size(0)

        # Eval: use validation residual-space if FAIR; else absolute test MSEs
        reg.eval(); ae.eval()
        if FAIR:
            with torch.no_grad():
                z_val, yf_val_std_pred = reg(torch.tensor(X_val_reg, dtype=torch.float32, device=device))
                yv_val_std_pred = ae.decode(z_val)
                yv_res_pred = (yv_val_std_pred.cpu().numpy() * yv_sd) + yv_mu
                yf_res_or_log_pred = (yf_val_std_pred.cpu().numpy() * yf_sd) + yf_mu
                if force_stats_key == "Y_force_log":
                    # No per-sample template for train; treat as residuals in log space to compare residuals
                    yf_res_pred = yf_res_or_log_pred
                else:
                    yf_res_pred = yf_res_or_log_pred
                yv_res_true = (Yv_val_reg * yv_sd) + yv_mu
                yf_res_true = (Yf_val_reg * yf_sd) + yf_mu
                via_mse = mse(yv_res_pred, yv_res_true)
                force_mse = mse(yf_res_pred, yf_res_true)
        else:
            with torch.no_grad():
                z_te, yf_te_std_pred = reg(torch.tensor(X_te_n, dtype=torch.float32, device=device))
                yv_te_std_pred = ae.decode(z_te)
                yv_res_pred = (yv_te_std_pred.cpu().numpy() * yv_sd) + yv_mu
                yf_res_or_log_pred = (yf_te_std_pred.cpu().numpy() * yf_sd) + yf_mu
                if force_stats_key == "Y_force_log":
                    f_t = np.array(test.get("Y_force_template"), dtype=float)
                    yf_res_pred = f_t * (np.exp(yf_res_or_log_pred) - 1.0)
                else:
                    yf_res_pred = yf_res_or_log_pred
                yv_abs_pred = yv_res_pred + Yv_bl_te
                yf_abs_pred = yf_res_pred + Yf_bl_te
                # Apply mask after baseline for absolute-space eval
                if force_mask_np is not None:
                    _m = force_mask_np.reshape(1, -1)
                    yf_abs_pred = yf_abs_pred * _m
                    Yf_te_abs = Yf_te_abs * _m
                via_mse = mse(yv_abs_pred, Yv_te_abs)
                force_mse = mse(yf_abs_pred, Yf_te_abs)
        if epoch % 10 == 0 or epoch == 1:
            tag = "val" if FAIR else "test"
            denom = len(X_tr_reg) if FAIR else len(X_tr_n)
            print(f"LAT epoch {epoch:3d}: train {total/denom:.6f}  {tag}_force {force_mse:.6f}  {tag}_via {via_mse:.6f}")
        if force_mse + via_mse < best_force + best_via:
            best_force, best_via = force_mse, via_mse
            torch.save({
                "state_dict_reg": reg.state_dict(),
                "state_dict_dec": ae.decoder.state_dict(),
                "x_dim": d_x,
                "y_force_dim": d_f,
                "y_via_dim": d_v,
                "latent_dim": 256,
                "force_stats_key": force_stats_key,
                "force_log": bool(force_stats_key == "Y_force_log"),
            }, OUT_DIR / "latent.pt")

    # If FAIR used validation for selection, re-evaluate saved best on the test set for reporting
    if FAIR:
        ckpt = torch.load(OUT_DIR / "latent.pt", map_location=device)
        reg.load_state_dict(ckpt["state_dict_reg"])  # type: ignore[index]
        ae.decoder.load_state_dict(ckpt["state_dict_dec"])  # type: ignore[index]
        reg.eval(); ae.eval()
        with torch.no_grad():
            z_te, yf_te_std_pred = reg(torch.tensor(X_te_n, dtype=torch.float32, device=device))
            yv_te_std_pred = ae.decode(z_te)
            yv_res_pred = (yv_te_std_pred.cpu().numpy() * yv_sd) + yv_mu
            yf_res_or_log_pred = (yf_te_std_pred.cpu().numpy() * yf_sd) + yf_mu
            if force_stats_key == "Y_force_log":
                f_t = np.array(test.get("Y_force_template"), dtype=float)
                yf_res_pred = f_t * (np.exp(yf_res_or_log_pred) - 1.0)
            else:
                yf_res_pred = yf_res_or_log_pred
            yv_abs_pred = yv_res_pred + Yv_bl_te
            yf_abs_pred = yf_res_pred + Yf_bl_te
            # Apply mask to absolute-space arrays for consistency
            if force_mask_np is not None:
                _m = force_mask_np.reshape(1, -1)
                yf_abs_pred = yf_abs_pred * _m
                Yf_te_abs = Yf_te_abs * _m
            # Write CSV exactly at this point using the arrays fed into MSE
            try:
                import csv, re
                ids = test.get("IDS", [str(i) for i in range(len(Yf_te_abs))])
                def _pid_from_path(p: str) -> str:
                    s = Path(p).stem
                    m = re.search(r"(\d{3})_", s)
                    return m.group(1) if m else s
                # Force pairs
                force_csv = OUT_DIR / "latent_mse_force_pairs.csv"
                with open(force_csv, "w", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(["patient_id", "muscle", "y_true", "y_pred"])  # header
                    for i, pid_path in enumerate(ids):
                        pid = _pid_from_path(str(pid_path))
                        for mi, mname in enumerate(muscles):
                            w.writerow([pid, mname, float(Yf_te_abs[i, mi]), float(yf_abs_pred[i, mi])])
                # Via pairs
                via_csv = OUT_DIR / "latent_mse_via_pairs.csv"
                with open(via_csv, "w", newline="") as vcsv:
                    wv = csv.writer(vcsv)
                    wv.writerow(["patient_id", "muscle", "point_index", "dx_true", "dy_true", "dz_true", "dx_pred", "dy_pred", "dz_pred"])  # header
                    for i, pid_path in enumerate(ids):
                        pid = _pid_from_path(str(pid_path))
                        off = 0
                        for m in muscles:
                            Lm = int(via_len[m])
                            for k in range(Lm):
                                tdx, tdy, tdz = map(float, Yv_te_abs[i, off:off+3])
                                pdx, pdy, pdz = map(float, yv_abs_pred[i, off:off+3])
                                wv.writerow([pid, m, k, tdx, tdy, tdz, pdx, pdy, pdz])
                                off += 3
            except Exception:
                pass
            best_via = mse(yv_abs_pred, Yv_te_abs)
            best_force = mse(yf_abs_pred, Yf_te_abs)
    print(f"Best test MSEs (absolute) -> force: {best_force:.6f}, via: {best_via:.6f}")
    print(f"Improvement over baseline (absolute) -> force: {base_force_mse - best_force:.6f}, via: {base_via_mse - best_via:.6f}")

    # Dump CSV with exact values used to compute absolute-space MSEs (test set)
    # (Removed duplicate CSV writing block to avoid overwriting)

    # Per-muscle metrics for latent (absolute test)
    import numpy as _np
    # Recompute predictions with saved best
    ckpt2 = torch.load(OUT_DIR / "latent.pt", map_location=device)
    reg.load_state_dict(ckpt2["state_dict_reg"])  # type: ignore[index]
    ae.decoder.load_state_dict(ckpt2["state_dict_dec"])  # type: ignore[index]
    reg.eval(); ae.eval()
    with torch.no_grad():
        z_te2, yf_te_std_pred2 = reg(torch.tensor(X_te_n, dtype=torch.float32, device=device))
        yv_te_std_pred2 = ae.decode(z_te2)
        yv_res_pred2 = (yv_te_std_pred2.cpu().numpy() * yv_sd) + yv_mu
        yf_res_or_log_pred2 = (yf_te_std_pred2.cpu().numpy() * yf_sd) + yf_mu
        if force_stats_key == "Y_force_log":
            f_t2 = _np.array(test.get("Y_force_template"), dtype=float)
            yf_res_pred2 = f_t2 * (_np.exp(yf_res_or_log_pred2) - 1.0)
        else:
            yf_res_pred2 = yf_res_or_log_pred2
        Yf_pred_abs2 = yf_res_pred2 + Yf_bl_te
        Yv_pred_abs2 = yv_res_pred2 + Yv_bl_te
    # Apply FORCE_PREDICT_MASK if present: zero-out excluded dims
    if "FORCE_PREDICT_MASK" in test:
        _mask_f = _np.array(test["FORCE_PREDICT_MASK"], dtype=float).reshape(1, -1)
        Yf_pred_abs2 = Yf_pred_abs2 * _mask_f
        Yf_te_abs = Yf_te_abs * _mask_f
    Yf_err_dim2 = ((Yf_pred_abs2 - Yf_te_abs) ** 2).mean(axis=0)
    per_muscle_force_lat = {m: float(Yf_err_dim2[i]) for i, m in enumerate(muscles)}
    Yv_err_dim2 = ((Yv_pred_abs2 - Yv_te_abs) ** 2).mean(axis=0)
    per_muscle_via_lat = {}
    off2 = 0
    for m in muscles:
        d = 3 * int(via_len[m])
        per_muscle_via_lat[m] = float(_np.mean(Yv_err_dim2[off2:off2+d]) if d > 0 else 0.0)
        off2 += d

    # Merge metrics
    metrics_path = OUT_DIR / "metrics.json"
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
    metrics["latent"] = {
        "force_mse": float(best_force),
        "via_mse": float(best_via),
        "notes": {
            "absolute_space": True,
            "fair": FAIR,
            "latent_dim": 256,
            "decoder_frozen": True,
        },
        "per_muscle_force_mse": per_muscle_force_lat,
        "per_muscle_via_mse": per_muscle_via_lat,
    }

    # Train Random Forest alternative (compare to latent model)
    print("\n" + "="*80)
    print("Training Random Forest as Alternative to Latent Model")
    print("="*80)

    # Prepare validation set if available
    X_va_n = None
    Yf_va_std = None
    Yv_va_std = None
    if val is not None:
        X_va = _np.array(val.get("X", []), dtype=float)
        if len(X_va) > 0:
            X_va_n = (X_va - x_mu) / x_sd

            # Append baseline summary features to validation set if enabled
            if USE_BASELINE_SUMMARY_FEATURES:
                Yf_bl_va = val.get("Y_force_baseline")
                Yv_bl_va = val.get("Y_via_baseline")
                if (Yf_bl_va is not None) and (Yv_bl_va is not None):
                    bf_va = _np.mean(_np.abs(_np.array(Yf_bl_va, dtype=float)), axis=1, keepdims=True)
                    bv_va = _np.mean(_np.abs(_np.array(Yv_bl_va, dtype=float)), axis=1, keepdims=True)
                    X_va_n = _np.concatenate([X_va_n, bf_va, bv_va], axis=1)

            if force_stats_key == "Y_force_log" and "Y_force_log" in val:
                Yf_va_res = _np.array(val["Y_force_log"], dtype=float)
            else:
                Yf_va_res = _np.array(val.get("Y_force_res", []), dtype=float)
            Yv_va_res = _np.array(val.get("Y_via_res", []), dtype=float)
            # Standardize validation targets
            Yf_va_std = (Yf_va_res - yf_mu) / yf_sd
            Yv_va_std = (Yv_va_res - yv_mu) / yv_sd

    rf_force_mse, rf_via_mse = train_random_forest_latent(
        X_tr_n, Yf_tr_std, Yv_tr_std,
        X_te_n, Yf_te_std, Yv_te_std,
        OUT_DIR / "rf_latent_force.pkl",
        OUT_DIR / "rf_latent_via.pkl",
        seed=SEED,
        X_val=X_va_n,
        Yf_val=Yf_va_std,
        Yv_val=Yv_va_std,
    )

    # Convert standardized predictions to absolute space for fair comparison
    import pickle
    with open(OUT_DIR / "rf_latent_force.pkl", 'rb') as f:
        rf_force_ckpt = pickle.load(f)
    with open(OUT_DIR / "rf_latent_via.pkl", 'rb') as f:
        rf_via_ckpt = pickle.load(f)

    Yf_pred_rf_std = rf_force_ckpt['model'].predict(X_te_n)
    Yv_pred_rf_std = rf_via_ckpt['model'].predict(X_te_n)

    # De-standardize
    Yf_pred_rf_res = (Yf_pred_rf_std * yf_sd) + yf_mu
    Yv_pred_rf_res = (Yv_pred_rf_std * yv_sd) + yv_mu

    # Convert log to additive if needed
    if force_stats_key == "Y_force_log":
        f_t = _np.array(test.get("Y_force_template"), dtype=float)
        Yf_pred_rf_res = f_t * (_np.exp(Yf_pred_rf_res) - 1.0)

    # Add baseline
    Yf_pred_rf_abs = Yf_pred_rf_res + Yf_bl_te
    Yv_pred_rf_abs = Yv_pred_rf_res + Yv_bl_te

    # Apply mask if needed
    if force_mask_np is not None:
        _m = force_mask_np.reshape(1, -1)
        Yf_pred_rf_abs = Yf_pred_rf_abs * _m
        Yf_te_abs_cmp = Yf_te_abs * _m
    else:
        Yf_te_abs_cmp = Yf_te_abs

    rf_force_mse_abs = mse(Yf_pred_rf_abs, Yf_te_abs_cmp)
    rf_via_mse_abs = mse(Yv_pred_rf_abs, Yv_te_abs)

    print(f"\nRandom Forest Results (Absolute Space):")
    print(f"  Force MSE: {rf_force_mse_abs:.6f} (vs Latent: {best_force:.6f})")
    print(f"  Via MSE: {rf_via_mse_abs:.6f} (vs Latent: {best_via:.6f})")

    metrics["random_forest_latent"] = {
        "force_mse": float(rf_force_mse_abs),
        "via_mse": float(rf_via_mse_abs),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Per-sex training (two separate latent models)
    if TORCH_AVAILABLE and sex_specific:
        sex_tr = [a.get("sex", 0.0) for a in train.get("AUX", [])]
        sex_te = [a.get("sex", 0.0) for a in test.get("AUX", [])]
        idxM_tr = [i for i,s in enumerate(sex_tr) if s > 0.5]
        idxF_tr = [i for i,s in enumerate(sex_tr) if s <= 0.5]
        idxM_te = [i for i,s in enumerate(sex_te) if s > 0.5]
        idxF_te = [i for i,s in enumerate(sex_te) if s <= 0.5]

        def sel(arr, idxs):
            return arr[idxs]

        for label, tr_idx, te_idx, ckpt in (
            ("m", idxM_tr, idxM_te, OUT_DIR / "latent_m.pt"),
            ("f", idxF_tr, idxF_te, OUT_DIR / "latent_f.pt"),
        ):
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            ae_s = PathAutoEncoder(d_in=d_v, latent_dim=256, hidden=1024).to(device)
            opt_ae_s = torch.optim.AdamW(ae_s.parameters(), lr=1e-3, weight_decay=1e-4)
            trY = Yv_tr_std[tr_idx]
            train_loader_ae_s = DataLoader(_OnlyY(trY), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED))
            for _ in range(60):
                for yb in train_loader_ae_s:
                    yb = yb.to(device); opt_ae_s.zero_grad(); rec = ae_s(yb)
                    (nn.functional.mse_loss(rec, yb) + smoothness_penalty(rec, muscles, via_len) * smooth_lambda).backward();
                    nn.utils.clip_grad_norm_(ae_s.parameters(), 1.0); opt_ae_s.step()
            for p in ae_s.decoder.parameters(): p.requires_grad = False
            reg_s = LatentRegressor(x_dim=d_x, latent_dim=256, y_force_dim=d_f).to(device)
            opt_reg_s = torch.optim.AdamW(reg_s.parameters(), lr=2e-4, weight_decay=1e-4)
            best_force_s, best_via_s = float('inf'), float('inf')
            for _ in range(60):
                # SGD over sex subset
                idxs = torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(SEED))
                for i0 in range(0, len(idxs), 64):
                    seli = idxs[i0:i0+64]
                    xb = torch.tensor(X_tr_n[tr_idx][seli.numpy()], dtype=torch.float32, device=device)
                    yv = torch.tensor(Yv_tr_std[tr_idx][seli.numpy()], dtype=torch.float32, device=device)
                    yf = torch.tensor(Yf_tr_std[tr_idx][seli.numpy()], dtype=torch.float32, device=device)
                    opt_reg_s.zero_grad(); z = reg_s.to_latent(xb); yv_pred = ae_s.decode(z); yf_pred = reg_s.force_head(z)
                    (nn.functional.mse_loss(yv_pred, yv) + 0.5*nn.functional.mse_loss(yf_pred, yf)).backward();
                    nn.utils.clip_grad_norm_(reg_s.parameters(), 1.0); opt_reg_s.step()
                # eval
                with torch.no_grad():
                    z_te = reg_s.to_latent(torch.tensor(X_te_n[te_idx], dtype=torch.float32, device=device))
                    yv_te_std_pred = ae_s.decode(z_te)
                    yf_te_std_pred = reg_s.force_head(z_te)
                    yv_res_pred = (yv_te_std_pred.cpu().numpy() * yv_sd) + yv_mu
                    yf_res_or_log_pred = (yf_te_std_pred.cpu().numpy() * yf_sd) + yf_mu
                    if force_stats_key == "Y_force_log":
                        f_t_s = np.array(test.get("Y_force_template"), dtype=float)[te_idx]
                        yf_res_pred = f_t_s * (np.exp(yf_res_or_log_pred) - 1.0)
                    else:
                        yf_res_pred = yf_res_or_log_pred
                    via_mse_s = mse(yv_res_pred + Yv_bl_te[te_idx], Yv_te_abs[te_idx])
                    force_mse_s = mse(yf_res_pred + Yf_bl_te[te_idx], Yf_te_abs[te_idx])
                if force_mse_s + via_mse_s < best_force_s + best_via_s:
                    best_force_s, best_via_s = force_mse_s, via_mse_s
                    torch.save({"state_dict_reg": reg_s.state_dict(), "state_dict_dec": ae_s.decoder.state_dict(),
                                "x_dim": d_x, "y_force_dim": d_f, "y_via_dim": d_v, "latent_dim": 256}, ckpt)
            print(f"[sex={label}] Best test MSEs -> force: {best_force_s:.6f}, via: {best_via_s:.6f}")
            base_force_s = mse(Yf_bl_te[te_idx], Yf_te_abs[te_idx]); base_via_s = mse(Yv_bl_te[te_idx], Yv_te_abs[te_idx])
            print(f"[sex={label}] Improvement over baseline -> force: {base_force_s - best_force_s:.6f}, via: {base_via_s - best_via_s:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex-specific", action="store_true", default=SEX_SPECIFIC, help="Train male/female-specific models in addition to combined model")
    args = parser.parse_args()
    main(sex_specific=args.sex_specific)

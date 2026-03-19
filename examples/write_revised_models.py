from __future__ import annotations

import json
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np

from osim_parser import OSIMModel
from repo_config import ANALYSIS_DIR, DATASETS_DIR, MODELS_DIR as REPO_MODELS_DIR, OUTPUTS_DIR, female_template_path, male_template_path, require_data_root

DATA_DIR = DATASETS_DIR.resolve()
MODELS_DIR = REPO_MODELS_DIR.resolve()
OUT_DIR = OUTPUTS_DIR.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== LEGACY TOGGLES (inherit from dataset build flags) ==========
# Controls whether to use sex-specific checkpoints when available (True) or
# a single sex-neutral checkpoint (False). We also write a comment with this
# setting into each output OSIM file.
USE_SEX_SPECIFIC = False

# ========== BASELINE SUMMARY FEATURES ==========
# Import from build_dataset.py to ensure consistency across all scripts
try:
    from examples.build_dataset import USE_BASELINE_SUMMARY_FEATURES, APPLY_VIA_PREDICT_MASK, APPLY_VIA_PREDICT_MASK_EXTRA
except ImportError:
    # Fallback if import fails
    USE_BASELINE_SUMMARY_FEATURES = False
    APPLY_VIA_PREDICT_MASK = True
    APPLY_VIA_PREDICT_MASK_EXTRA = True

# Simple in-code toggles: set True/False to include/exclude models without CLI flags.
# You can comment/uncomment these to control which models write outputs.
ENABLE_MLP = True
ENABLE_TRANSFORMER = False
ENABLE_GNN = False
ENABLE_LATENT = True
ENABLE_RANDOM_FOREST = False  # Public repo does not bundle RF checkpoints by default
ENABLE_BASELINE = False
ENABLE_AGE_ADJUSTED_BASELINE = False  # Age-adjusted generic template


def _load_build_flags():
    """Load build flags from stats.json to ensure consistency."""
    try:
        with open(DATA_DIR / "stats.json", "r") as f:
            stats = json.load(f)
        flags = stats.get("_build_flags", {})
        return flags
    except Exception:
        return {}


def load_age_force_relationships():
    """Load age-force linear regression results from analysis."""
    analysis_dir = ANALYSIS_DIR.resolve()
    try:
        with open(analysis_dir / "age_force_results.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load age-force results: {e}")
        return None


def get_muscle_group_prefix(muscle_name: str) -> str | None:
    """Determine which muscle group a muscle belongs to based on prefix."""
    prefixes = {
        "Psoas (Ps_)": "Ps_",
        "Iliocostalis (IL_)": "IL_",
        "Longissimus (LT)": "LT",
        "Quadratus Lumborum (QL_)": "QL_",
        "Multifidus (multifidus_)": "multifidus_",
        "Erector Spinae (E0_)": "E0_",
        "Trapezius (trap_)": "trap_",
        "Latissimus Dorsi (LD_)": "LD_",
    }
    for group_name, prefix in prefixes.items():
        if muscle_name.startswith(prefix):
            return group_name
    return None


def compute_age_adjusted_force(template_force: float, age: float, sex: str,
                                muscle_name: str, age_force_data: Dict) -> float:
    """Compute age-adjusted max isometric force.

    Args:
        template_force: Template max isometric force (at reference age 65)
        age: Subject age in years
        sex: 'M' or 'F'
        muscle_name: Name of the muscle
        age_force_data: Dictionary with age-force regression results

    Returns:
        Age-adjusted force value
    """
    if age_force_data is None:
        return template_force

    # Reference age (template represents this age)
    reference_age = 65.0

    # Find muscle group
    group_name = get_muscle_group_prefix(muscle_name)
    if group_name is None:
        return template_force

    # Get regression parameters
    group_data = age_force_data.get(group_name)
    if group_data is None:
        return template_force

    sex_key = "male" if sex == "M" else "female"
    sex_data = group_data.get(sex_key)
    if sex_data is None:
        return template_force

    slope = sex_data.get("slope", 0.0)

    # Adjustment: force_adjusted = force_template + (age - reference_age) * slope
    age_diff = age - reference_age
    adjusted_force = template_force + (age_diff * slope)

    # Don't allow negative forces
    return max(0.0, adjusted_force)


def load_checkpoint(path: Path):
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt


def predict(model_ckpt, X: List[List[float]]):
    import torch
    from examples.train_models import MLP  # reuse architecture
    d_in = model_ckpt.get("d_in") if isinstance(model_ckpt, dict) else None
    d_out = model_ckpt.get("d_out") if isinstance(model_ckpt, dict) else None
    state = model_ckpt.get("state_dict") if isinstance(model_ckpt, dict) and ("state_dict" in model_ckpt) else model_ckpt
    if d_in is None or d_out is None:
        # Infer shapes from state dict
        w0 = state.get('net.0.weight') or next(iter(state.values()))
        d_out = int(list(state.values())[-1].shape[0])
        d_in = int(w0.shape[1])
    net = MLP(d_in, d_out)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        Y = net(X_t).cpu().numpy()
    # De-standardize residuals if FAIR stats present in checkpoint
    y_mu = model_ckpt.get("y_mu") if isinstance(model_ckpt, dict) else None
    y_sd = model_ckpt.get("y_sd") if isinstance(model_ckpt, dict) else None
    if y_mu is not None and y_sd is not None:
        import numpy as _np
        Y = (Y * _np.array(y_sd, dtype=float)) + _np.array(y_mu, dtype=float)
    return Y


def predict_random_forest(model_path: Path, X: List[List[float]]):
    """Predict using Random Forest model.

    Args:
        model_path: Path to pickled Random Forest model (.pkl file)
        X: List of feature vectors (already normalized)

    Returns:
        Y: Predicted residuals (de-standardized if stats were stored)
    """
    import pickle
    import numpy as np

    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    rf_model = ckpt['model']
    Y = rf_model.predict(X)

    # Note: Random Forest models from train_models.py receive already-standardized
    # inputs and produce standardized outputs, but we don't store y_mu/y_sd in the
    # pickle file. The calling code should handle de-standardization if needed.
    return Y


def predict_transformer(ckpt_path: Path, X_seq, X_seq_mask, muscles: List[str], via_len: Dict[str, int], stats: Dict):
    """Return model-predicted residuals for force and via using transformer.

    - Inputs X_seq are normalized with stats["X_seq"].
    - Model outputs standardized residuals; we de-standardize using residual stats.
    - If FORCE_LOG was used for force during training (stats contains Y_force_log),
      return additive residuals by converting from log space using Y_force_template at call site.
    """
    import torch
    import torch.nn as nn
    import numpy as np

    try:
        ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt_raw if isinstance(ckpt_raw, dict) and ("state_dict" not in ckpt_raw) else ckpt_raw.get("state_dict")
        meta = {} if state is ckpt_raw else ckpt_raw
    except TypeError:
        ckpt_raw = torch.load(ckpt_path, map_location="cpu")
        state = ckpt_raw.get("state_dict") or ckpt_raw
        meta = ckpt_raw if ("state_dict" in ckpt_raw) else {}

    # Infer shapes if metadata missing
    def _infer_num_layers(sd: Dict[str, torch.Tensor]) -> int:
        mx = -1
        for k in sd.keys():
            if k.startswith('encoder.layers.'):
                try:
                    idx = int(k.split('.')[2])
                    mx = max(mx, idx)
                except Exception:
                    pass
        return (mx + 1) if mx >= 0 else 4

    import torch as _torch
    proj_w = state.get('proj_in.weight', None)
    if proj_w is None:
        # fallback: try module prefix
        for k in state.keys():
            if k.endswith('proj_in.weight'):
                proj_w = state[k]
                break
    d_model = int(meta.get("d_model") or (proj_w.shape[0] if proj_w is not None else 256))
    token_dim = int(meta.get("token_dim") or (proj_w.shape[1] if proj_w is not None else (len(X_seq[0][0]) )))
    nhead = int(meta.get("nhead") or 8)
    num_layers = int(meta.get("num_layers") or _infer_num_layers(state))
    dim_ff_w = state.get('encoder.layers.0.linear1.weight', None)
    dim_ff = int(meta.get("dim_feedforward") or (dim_ff_w.shape[0] if dim_ff_w is not None else 512))
    token_count = int(meta.get("token_count") or len(X_seq[0]))

    per_muscle_via_dims = [3 * int(via_len[m]) for m in muscles]
    T = token_count

    def sinusoidal_pos_encoding(length: int, d_model: int):
        import math
        pe = [[0.0] * d_model for _ in range(length)]
        for pos in range(length):
            for i in range(0, d_model, 2):
                div = math.pow(10000.0, i / d_model)
                pe[pos][i] = math.sin(pos / div)
                if i + 1 < d_model:
                    pe[pos][i + 1] = math.cos(pos / div)
        return pe

    class TransformerModel(nn.Module):
        def __init__(self, token_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int,
                     out_force: int, out_via: int, token_count: int, token_dropout: float = 0.0):
            super().__init__()
            self.proj_in = nn.Linear(token_dim, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.cls = nn.Parameter(_torch.zeros(1, 1, d_model))
            self.drop_tok = nn.Dropout(p=token_dropout)
            self.pos_enc = nn.Parameter(_torch.tensor(sinusoidal_pos_encoding(token_count + 1, d_model), dtype=_torch.float32), requires_grad=False)
            self.head_force = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_force))
            self.head_via = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_via))

    class TransformerWithQueries(TransformerModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            d_model_l = self.pos_enc.shape[-1]
            dec_layer = nn.TransformerDecoderLayer(d_model=d_model_l, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
            self.muscle_queries = nn.Parameter(_torch.zeros(1, len(muscles), d_model_l))
            nn.init.xavier_uniform_(self.muscle_queries)
            self.ln_force = nn.LayerNorm(d_model_l)
            self.ln_via = nn.LayerNorm(d_model_l)
            self.force_head_shared = nn.Linear(d_model_l, 1)
            # Build via heads only for muscles with >0 via dims to avoid zero-width layers
            self.via_idx = [i for i, d in enumerate(per_muscle_via_dims) if d > 0]
            self.via_heads = nn.ModuleList([nn.Linear(d_model_l, per_muscle_via_dims[i]) for i in self.via_idx])
        def forward(self, x, mask, token_types):
            B, Tloc, F = x.shape
            device = next(self.parameters()).device
            tt = _torch.tensor(token_types, device=device).unsqueeze(0).expand(B, -1)
            ti = _torch.arange(Tloc, device=device).unsqueeze(0).expand(B, -1)
            x_proj = self.proj_in(x)
            enc_in = _torch.cat([self.cls.expand(B, -1, -1), x_proj], dim=1)
            pos = self.pos_enc[:Tloc+1, :].unsqueeze(0).expand(B, -1, -1)
            enc_in = self.drop_tok(enc_in + pos)
            key_padding = _torch.zeros(B, Tloc + 1, dtype=_torch.bool, device=device)
            key_padding[:, 1:] = (mask <= 0.5)
            memory = self.encoder(enc_in, src_key_padding_mask=key_padding)
            queries = self.muscle_queries.expand(B, -1, -1)
            dec_out = self.decoder(tgt=queries, memory=memory)
            y_force = self.force_head_shared(self.ln_force(dec_out)).squeeze(-1)
            via_chunks = []
            h_v = self.ln_via(dec_out)
            for j, i in enumerate(self.via_idx):
                via_chunks.append(self.via_heads[j](h_v[:, i, :]))
            y_via = _torch.cat(via_chunks, dim=-1) if len(via_chunks) > 0 else _torch.zeros(B, 0, device=device)
            return y_force, y_via

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = TransformerWithQueries(token_dim=token_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                   dim_feedforward=dim_ff, out_force=len(muscles), out_via=sum(per_muscle_via_dims),
                                   token_count=T, token_dropout=0.0).to(device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    model.eval()

    # Token types derived from SEQ_TOKENS ordering in dataset
    seq_tokens = stats.get("SEQ_TOKENS")
    if seq_tokens is None:
        # fallback: from current data (assumes same across split files)
        with open(DATA_DIR / "test.json", "r") as _f:
            _d = json.load(_f)
        seq_tokens = _d.get("SEQ_TOKENS")
    token_types = []
    for name in seq_tokens:
        tt = 0 if str(name).startswith("lumbar") else (1 if str(name).startswith("thoracic") else 2)
        token_types.append(int(tt))

    # Normalize inputs
    mu = np.array(stats.get("X_seq", {}).get("mean", []), dtype=float)
    sd = np.array(stats.get("X_seq", {}).get("std", []), dtype=float)
    Xs = (np.array(X_seq, dtype=float) - mu) / sd
    Ms = np.array(X_seq_mask, dtype=float)

    # Predict in batches
    bs = 64
    outs_f = []
    outs_v = []
    with _torch.no_grad():
        for i in range(0, len(Xs), bs):
            xb = _torch.tensor(Xs[i:i+bs], dtype=_torch.float32, device=device)
            mb = _torch.tensor(Ms[i:i+bs], dtype=_torch.float32, device=device)
            yf, yv = model(xb, mb, token_types)
            outs_f.append(yf.cpu().numpy())
            outs_v.append(yv.cpu().numpy())
    Yf_std = np.concatenate(outs_f, axis=0)
    Yv_std = np.concatenate(outs_v, axis=0)

    # De-standardize residuals (force may be log-standardized)
    force_stats_key = str((meta.get("force_stats_key") if isinstance(meta, dict) else None) or ("Y_force_log" if ("Y_force_log" in stats) else "Y_force_res"))
    mu_f = np.array(stats.get(force_stats_key, {}).get("mean", []), dtype=float)
    sd_f = np.array(stats.get(force_stats_key, {}).get("std", []), dtype=float)
    mu_v = np.array(stats.get("Y_via_res", {}).get("mean", []), dtype=float)
    sd_v = np.array(stats.get("Y_via_res", {}).get("std", []), dtype=float)
    Yf_res = (Yf_std * sd_f) + mu_f  # if use_force_log=True, this is log-residual; caller may convert
    Yv_res = (Yv_std * sd_v) + mu_v

    return Yf_res, Yv_res


def predict_latent(ckpt_path: Path, X: List[List[float]], stats: Dict):
    """Use latent model to predict residuals for force and via.

    - Normalize X by stats["X"].
    - Model returns standardized residuals; we de-standardize using residual stats.
    - If force was trained in log space (stats contains Y_force_log), caller converts to additive via Y_force_template.
    """
    import torch
    import torch.nn as nn
    import numpy as np

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    x_mu = np.array(stats.get("X", {}).get("mean", []), dtype=float)
    x_sd = np.array(stats.get("X", {}).get("std", []), dtype=float)
    Xn = (np.array(X, dtype=float) - x_mu) / x_sd

    x_dim = int(ckpt.get("x_dim", Xn.shape[1])) ; latent_dim = int(ckpt.get("latent_dim", 192)) ; y_f_dim = int(ckpt.get("y_force_dim",  len(stats.get("Y_force_res",{}).get("mean",[])))) ; y_v_dim = int(ckpt.get("y_via_dim", len(stats.get("Y_via_res",{}).get("mean",[]))) )

    class Decoder(nn.Module):
        def __init__(self, latent_dim: int, y_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, y_dim),
            )
        def forward(self, z):
            return self.net(z)

    class Regressor(nn.Module):
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
        def forward(self, x):
            z = self.to_latent(x)
            y_force = self.force_head(z)
            return z, y_force

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = Decoder(latent_dim=latent_dim, y_dim=y_v_dim).to(device)
    reg = Regressor(x_dim=x_dim, latent_dim=latent_dim, y_force_dim=y_f_dim).to(device)
    # Load
    state_dec = ckpt.get("state_dict_dec") if isinstance(ckpt, dict) else None
    state_reg = ckpt.get("state_dict_reg") if isinstance(ckpt, dict) else None
    if state_dec:
        dec.load_state_dict(state_dec, strict=False)
    if state_reg:
        reg.load_state_dict(state_reg, strict=False)
    dec.eval() ; reg.eval()

    # Predict
    bs = 128
    out_v_std = [] ; out_f_std = []
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            xb = torch.tensor(Xn[i:i+bs], dtype=torch.float32, device=device)
            z, yf_std = reg(xb)
            yv_std = dec(z)
            out_v_std.append(yv_std.cpu().numpy())
            out_f_std.append(yf_std.cpu().numpy())
    Yv_std = np.concatenate(out_v_std, axis=0)
    Yf_std = np.concatenate(out_f_std, axis=0)

    # De-standardize residuals (force may be log-standardized)
    mu_v = np.array(stats.get("Y_via_res", {}).get("mean", []), dtype=float)
    sd_v = np.array(stats.get("Y_via_res", {}).get("std", []), dtype=float)
    use_force_log = ("Y_force_log" in stats)
    mu_f = np.array(stats.get("Y_force_log" if use_force_log else "Y_force_res", {}).get("mean", []), dtype=float)
    sd_f = np.array(stats.get("Y_force_log" if use_force_log else "Y_force_res", {}).get("std", []), dtype=float)
    Yv_res = (Yv_std * sd_v) + mu_v
    Yf_res = (Yf_std * sd_f) + mu_f  # if force_stats_key==Y_force_log, this is log residual
    return Yf_res, Yv_res


def predict_gnn(ckpt_path: Path, X_seq, X_seq_mask, stats: Dict, muscles: List[str], via_len: Dict[str, int]):
    """Run SpineGNN checkpoint to predict residuals (not standardized)."""
    import torch
    import torch.nn as nn
    import numpy as np

    try:
        ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt_raw if isinstance(ckpt_raw, dict) and ("state_dict" not in ckpt_raw) else ckpt_raw.get("state_dict")
        meta = {} if state is ckpt_raw else ckpt_raw
    except TypeError:
        ckpt_raw = torch.load(ckpt_path, map_location="cpu")
        state = ckpt_raw.get("state_dict") or ckpt_raw
        meta = ckpt_raw if ("state_dict" in ckpt_raw) else {}

    # Infer dims from state if metadata missing
    w0 = state.get('layers.0.lin_self.weight')
    token_dim = int(meta.get("token_dim") or (w0.shape[1] if w0 is not None else (len(X_seq[0][0]) * 4)))
    hidden = int(meta.get("hidden") or (w0.shape[0] if w0 is not None else 256))
    # count layers
    def _count_layers(sd):
        mx = -1
        for k in sd.keys():
            if k.startswith('layers.'):
                try:
                    idx = int(k.split('.')[1])
                    mx = max(mx, idx)
                except Exception:
                    pass
        return (mx + 1) if mx >= 0 else 4
    num_layers = int(meta.get("num_layers") or _count_layers(state))

    class GraphConv(nn.Module):
        def __init__(self, d_in: int, d_out: int):
            super().__init__()
            self.lin_self = nn.Linear(d_in, d_out)
            self.lin_nei = nn.Linear(d_in, d_out)
            self.act = nn.ReLU()
        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h_self = self.lin_self(x)
            left = torch.roll(x, 1, dims=1); left[:, 0, :] = 0.0
            right = torch.roll(x, -1, dims=1); right[:, -1, :] = 0.0
            h_nei = self.lin_nei(left + right)
            h = (h_self + h_nei) * mask.unsqueeze(-1)
            return self.act(h)

    class SpineGNN(nn.Module):
        def __init__(self, token_dim: int, hidden: int, num_layers: int, muscles: List[str], via_len: Dict[str, int]):
            super().__init__()
            self.layers = nn.ModuleList([GraphConv(token_dim if i == 0 else hidden, hidden) for i in range(num_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
            self.readout_node = nn.Linear(hidden, hidden)
            self.force_head = nn.Linear(hidden, len(muscles))
            self.via_heads = nn.ModuleList([nn.Linear(hidden, 3*via_len[m]) for m in muscles])
        def forward(self, x: torch.Tensor, mask: torch.Tensor):
            h = x
            for gc, ln in zip(self.layers, self.norms):
                h = gc(h, mask)
                h = ln(h)
            h_node = self.readout_node(h)
            g = (h_node * mask.unsqueeze(-1)).mean(dim=1)
            y_force = self.force_head(g)
            y_via = torch.cat([head(g) for head in self.via_heads], dim=-1) if len(self.via_heads) > 0 else torch.zeros(g.size(0), 0, device=g.device)
            return y_force, y_via

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpineGNN(token_dim=token_dim, hidden=hidden, num_layers=num_layers, muscles=muscles, via_len=via_len).to(device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    model.eval()

    mu = np.array(stats.get("X_seq", {}).get("mean", []), dtype=float)
    sd = np.array(stats.get("X_seq", {}).get("std", []), dtype=float)
    Xs_raw = (np.array(X_seq, dtype=float) - mu) / sd
    # augment same as training
    left = np.roll(Xs_raw, 1, axis=1); left[:, 0, :] = 0.0
    right = np.roll(Xs_raw, -1, axis=1); right[:, -1, :] = 0.0
    d_prev = Xs_raw - left
    d_next = right - Xs_raw
    curvature = left - 2*Xs_raw + right
    Xs = np.concatenate([Xs_raw, d_prev, d_next, curvature], axis=-1)
    Ms = np.array(X_seq_mask, dtype=float)
    bs = 64
    outs_f = []
    outs_v = []
    with torch.no_grad():
        for i in range(0, len(Xs), bs):
            xb = torch.tensor(Xs[i:i+bs], dtype=torch.float32, device=device)
            mb = torch.tensor(Ms[i:i+bs], dtype=torch.float32, device=device)
            yf, yv = model(xb, mb)
            outs_f.append(yf.cpu().numpy())
            outs_v.append(yv.cpu().numpy())
    Yf_res = np.concatenate(outs_f, axis=0)
    Yv_res = np.concatenate(outs_v, axis=0)
    return Yf_res, Yv_res


def _load_split(split: str):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    with open(DATA_DIR / f"{split}.json", "r") as f:
        d = json.load(f)
    return d


def _concat(a, b):
    if a is None:
        return b
    return a + b


def _height_estimate(model: OSIMModel) -> float:
    """Estimate subject height (meters) from model kinematic chain without clamping.

    Uses Y-translation of head_neck relative to sacrum if available, otherwise thoracic12,
    otherwise thoracic5, otherwise falls back to 1.75.
    """
    root = "sacrum"
    q = {}
    bodies = model.data.get("bodies", {})
    target = None
    for cand in ("head_neck", "thoracic12", "thoracic5"):
        if cand in bodies:
            target = cand
            break
    if target is None:
        return 1.75
    X = model.transform_relative_to(target, root, q)
    return float(X[1][3])


def _sacrum_mass_ratio(subject: OSIMModel, template: OSIMModel) -> float:
    try:
        ms = float(subject.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
        mt = float(template.data["bodies"]["sacrum"]["mass"])  # type: ignore[index]
        return ms / mt if mt else 1.0
    except Exception:
        return 1.0


def _get_pathpoint_x(model: OSIMModel, muscle_name: str, point_name: str) -> float | None:
    try:
        m = model.data.get("forces", {}).get(muscle_name) or {}
        pts = m.get("path_points") or []
        for pt in pts:
            nm = pt.get("name")
            if isinstance(nm, str) and (nm == point_name):
                loc = pt.get("location") or [0.0, 0.0, 0.0]
                return float(loc[0])
    except Exception:
        pass
    return None


def _corb_height_ratio_via_x(subject: OSIMModel, template: OSIMModel) -> float | None:
    """Estimate scale using CORB.PathPoint_scapula_R x-coordinate ratio.

    Returns None if unavailable; caller should fall back to legacy estimate.
    """
    subj_x = _get_pathpoint_x(subject, "CORB", "PathPoint_scapula_R")
    tmpl_x = _get_pathpoint_x(template, "CORB", "PathPoint_scapula_R")
    if (subj_x is not None) and (tmpl_x is not None):
        eps = 1e-8
        if abs(tmpl_x) > eps:
            return float(subj_x / tmpl_x)
    return None


def scale_muscle_fiber_properties(model: OSIMModel, muscles: List[str], template_props: Dict[str, Dict], sex: str) -> None:
    """Scale optimal_fiber_length and tendon_slack_length based on muscle length ratio.

    Args:
        model: The OSIMModel to modify (via points should already be updated)
        muscles: List of muscle names to process
        template_props: Dict from cached JSON with template properties
        sex: "male" or "female" to select correct template
    """
    template_data = template_props.get(sex, {})

    for muscle_name in muscles:
        try:
            # Get template properties
            tmpl_muscle = template_data.get(muscle_name)
            if not tmpl_muscle:
                continue

            tmpl_length = tmpl_muscle.get("length")
            tmpl_ofl = tmpl_muscle.get("optimal_fiber_length")
            tmpl_tsl = tmpl_muscle.get("tendon_slack_length")

            # Skip if any template property is None or zero
            if not tmpl_length or tmpl_length <= 0.0:
                continue
            if tmpl_ofl is None or tmpl_tsl is None:
                continue

            # Calculate new muscle length from updated via points
            new_length = model.calculate_muscle_length(muscle_name)

            # Calculate scaling ratio
            ratio = new_length / tmpl_length

            # Apply scaling
            new_ofl = tmpl_ofl * ratio
            new_tsl = tmpl_tsl * ratio

            model.set_muscle_optimal_fiber_length(muscle_name, new_ofl)
            model.set_muscle_tendon_slack_length(muscle_name, new_tsl)

        except Exception as e:
            # Silently skip muscles that fail (e.g., missing properties)
            pass


def scale_body_masses(model: OSIMModel, w_ratio: float) -> None:
    """Scale all body masses by weight ratio.

    Args:
        model: The OSIMModel to modify
        w_ratio: Weight scaling factor (subject_weight / template_weight)
    """
    try:
        for body_name in model.data.get("bodies", {}).keys():
            body_data = model.data["bodies"][body_name]
            original_mass = body_data.get("mass")
            if original_mass is not None and isinstance(original_mass, (int, float)):
                new_mass = float(original_mass) * float(w_ratio)
                model.set_body_mass(body_name, new_mass)
    except Exception as e:
        # Silently skip if scaling fails
        pass


def main():
    # Load build flags from dataset to ensure consistency
    build_flags = _load_build_flags()
    VIA_MILLIMETERS = build_flags.get("VIA_MILLIMETERS", True)
    CORB_HEIGHT_SCALING = build_flags.get("CORB_HEIGHT_SCALING", True)
    APPLY_FORCE_PREDICT_MASK = build_flags.get("APPLY_FORCE_PREDICT_MASK", True)
    POS_ONLY = build_flags.get("POS_ONLY", False)
    FORCE_BASELINE_SCALING = build_flags.get("FORCE_BASELINE_SCALING", True)
    FAIR = build_flags.get("FAIR", True)
    FORCE_LOG = build_flags.get("FORCE_LOG", False)

    print(f"Writing models with dataset flags:")
    print(f"  VIA_MILLIMETERS={VIA_MILLIMETERS}")
    print(f"  CORB_HEIGHT_SCALING={CORB_HEIGHT_SCALING}")
    print(f"  APPLY_FORCE_PREDICT_MASK={APPLY_FORCE_PREDICT_MASK}")
    print(f"  POS_ONLY={POS_ONLY}")
    print(f"  FORCE_BASELINE_SCALING={FORCE_BASELINE_SCALING}")

    # Helper function to convert via deltas based on flag
    def via_to_meters(delta_mm_or_m):
        """Convert via delta to meters based on VIA_MILLIMETERS flag."""
        if VIA_MILLIMETERS:
            return float(delta_mm_or_m) / 1000.0  # millimeters to meters
        else:
            return float(delta_mm_or_m)  # already in meters

    parser = argparse.ArgumentParser(description="Write baseline and revised OSIMs from dataset and models")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test", help="Which split to export")
    parser.add_argument("--models", type=str, default="", help="Comma-separated list among: mlp, transformer, gnn, latent, rf (optional)")
    args = parser.parse_args()

    if args.models:
        models_to_use = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    else:
        models_to_use = set()
        if ENABLE_MLP: models_to_use.add("mlp")
        if ENABLE_TRANSFORMER: models_to_use.add("transformer")
        if ENABLE_GNN: models_to_use.add("gnn")
        if ENABLE_LATENT: models_to_use.add("latent")
        if ENABLE_RANDOM_FOREST: models_to_use.add("rf")

    # Load dataset (selected splits)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    data = None
    for sp in splits:
        d = _load_split(sp)
        if data is None:
            data = d
        else:
            # Concatenate fields across splits
            for k in ["X", "X_seq", "X_seq_mask", "IDS", "Y_force_baseline", "Y_via_baseline"]:
                data[k] = _concat(data[k], d[k])
            # MUSCLES and VIA_LEN must be consistent; keep first
    assert data is not None

    X = data["X"]
    IDS = data["IDS"]
    muscles = data["MUSCLES"]
    via_len = data["VIA_LEN"]

    # Load stats (for normalization) and models
    stats = {}
    try:
        with open(DATA_DIR / "stats.json", "r") as sf:
            stats = json.load(sf)
    except Exception:
        stats = {}
    # Store sequence token names to stats so predictor can use them
    if "SEQ_TOKENS" in data:
        stats["SEQ_TOKENS"] = data["SEQ_TOKENS"]

    # Load cached template muscle properties
    template_props = {}
    try:
        with open(DATA_DIR / "generic_muscle_properties.json", "r") as tmpf:
            template_props = json.load(tmpf)
        print(f"Loaded template muscle properties for {len(template_props.get('muscles', []))} muscles")
    except Exception as e:
        print(f"Warning: Could not load template muscle properties: {e}")
        template_props = {}

    # Load age-force linear relationships for age-adjusted baseline
    age_force_data = None
    if ENABLE_AGE_ADJUSTED_BASELINE:
        age_force_data = load_age_force_relationships()
        if age_force_data is not None:
            print(f"Loaded age-force relationships for {len(age_force_data)} muscle groups")

    ckpt_force = ckpt_via = None

    Xs = data.get("X_seq"); Ms = data.get("X_seq_mask")

    # Transformer predictions (optional). Support sex-specific ckpts.
    dF_pred_transf = dVia_pred_transf = None
    if ("transformer" in models_to_use) and (Xs is not None) and (Ms is not None):
        try:
            idx_male = [i for i, p in enumerate(data["IDS"]) if "/Male/" in p]
            idx_fem = [i for i, p in enumerate(data["IDS"]) if "/Female/" in p]
            def sel(lst, ids):
                return [lst[i] for i in ids]
            out_f = np.zeros((len(Xs), len(muscles)), dtype=float)
            out_v = np.zeros((len(Xs), sum(3*via_len[m] for m in muscles)), dtype=float)
            did_any = False
            if USE_SEX_SPECIFIC and (MODELS_DIR / "transf_seq_m.pt").exists() and (MODELS_DIR / "transf_seq_f.pt").exists():
                if idx_male:
                    Yf_m, Yv_m = predict_transformer(MODELS_DIR / "transf_seq_m.pt", sel(Xs, idx_male), sel(Ms, idx_male), muscles, via_len, stats)
                    out_f[idx_male, :] = Yf_m; out_v[idx_male, :] = Yv_m; did_any = True
                if idx_fem:
                    Yf_f, Yv_f = predict_transformer(MODELS_DIR / "transf_seq_f.pt", sel(Xs, idx_fem), sel(Ms, idx_fem), muscles, via_len, stats)
                    out_f[idx_fem, :] = Yf_f; out_v[idx_fem, :] = Yv_f; did_any = True
            else:
                ckpt = MODELS_DIR / "transf_seq.pt"
                if ckpt.exists():
                    Yf_all, Yv_all = predict_transformer(ckpt, Xs, Ms, muscles, via_len, stats)
                    out_f[:, :] = Yf_all; out_v[:, :] = Yv_all; did_any = True
            if did_any:
                # If force residuals are in log space, convert to additive using template forces
                if "Y_force_log" in stats and ("Y_force_template" in data):
                    f_t_all = np.array(data["Y_force_template"], dtype=float)
                    out_f_add = f_t_all * (np.exp(out_f) - 1.0)
                else:
                    out_f_add = out_f
                # Apply FORCE_PREDICT_MASK to residuals first, then add baseline
                if "FORCE_PREDICT_MASK" in data:
                    _mask = np.array(data["FORCE_PREDICT_MASK"], dtype=float).reshape(1, -1)
                    out_f_add = out_f_add * _mask
                # Apply VIA_PREDICT_MASK to residuals
                # Note: Masked residuals become zero, resulting in height-scaled via points:
                # P_final = P_template + P_baseline + 0 = P_template * h_ratio (pure height scaling)
                if APPLY_VIA_PREDICT_MASK and "FORCE_PREDICT_MASK" in data:
                    force_mask = data["FORCE_PREDICT_MASK"]
                    via_mask_list = []
                    for mi, m in enumerate(muscles):
                        L = int(via_len[m])
                        mask_val = float(force_mask[mi])
                        via_mask_list.extend([mask_val] * (3 * L))
                    # Apply extra mask if enabled
                    if APPLY_VIA_PREDICT_MASK_EXTRA and "VIA_PREDICT_MASK_EXTRA" in data:
                        via_mask_extra = data["VIA_PREDICT_MASK_EXTRA"]
                        via_mask_combined = []
                        for mi, m in enumerate(muscles):
                            L = int(via_len[m])
                            combined_mask = float(force_mask[mi]) * float(via_mask_extra[mi])
                            via_mask_combined.extend([combined_mask] * (3 * L))
                        via_mask_list = via_mask_combined
                    _via_mask = np.array(via_mask_list, dtype=float).reshape(1, -1)
                    out_v = out_v * _via_mask
                dF_abs = out_f_add + np.array(data["Y_force_baseline"])
                dF_pred_transf = dF_abs
                dVia_pred_transf = (out_v + np.array(data["Y_via_baseline"]))
        except Exception as e:
            print("Warning: Transformer inference failed; skipping transformer revised outputs:", e)

    # Latent model predictions (optional). Support sex-specific ckpts.
    dF_pred_lat = dVia_pred_lat = None
    if "latent" in models_to_use:
        try:
            # Prepare input features (optionally with baseline summaries)
            X_for_latent = X
            if USE_BASELINE_SUMMARY_FEATURES:
                # Append baseline summary features to match training
                Yf_bl_all = np.array(data["Y_force_baseline"], dtype=float)
                Yv_bl_all = np.array(data["Y_via_baseline"], dtype=float)
                bf_all = np.mean(np.abs(Yf_bl_all), axis=1, keepdims=True)
                bv_all = np.mean(np.abs(Yv_bl_all), axis=1, keepdims=True)
                # X is a list of lists, need to convert and augment
                X_for_latent = [list(X[i]) + [float(bf_all[i,0]), float(bv_all[i,0])] for i in range(len(X))]

            idx_male = [i for i, p in enumerate(data["IDS"]) if "/Male/" in p]
            idx_fem = [i for i, p in enumerate(data["IDS"]) if "/Female/" in p]
            out_f_l = np.zeros((len(X), len(muscles)), dtype=float)
            out_v_l = np.zeros((len(X), sum(3*via_len[m] for m in muscles)), dtype=float)
            did_any_l = False
            if USE_SEX_SPECIFIC and (MODELS_DIR / "latent_m.pt").exists() and (MODELS_DIR / "latent_f.pt").exists():
                if idx_male:
                    Yf_m, Yv_m = predict_latent(MODELS_DIR / "latent_m.pt", [X_for_latent[i] for i in idx_male], stats)
                    out_f_l[idx_male, :] = Yf_m; out_v_l[idx_male, :] = Yv_m; did_any_l = True
                if idx_fem:
                    Yf_f, Yv_f = predict_latent(MODELS_DIR / "latent_f.pt", [X_for_latent[i] for i in idx_fem], stats)
                    out_f_l[idx_fem, :] = Yf_f; out_v_l[idx_fem, :] = Yv_f; did_any_l = True
            else:
                latent_ckpt = MODELS_DIR / "latent.pt"
                if latent_ckpt.exists():
                    Yf_all, Yv_all = predict_latent(latent_ckpt, X_for_latent, stats)
                    out_f_l[:, :] = Yf_all; out_v_l[:, :] = Yv_all; did_any_l = True
            if did_any_l:
                if "Y_force_log" in stats and ("Y_force_template" in data):
                    f_t_all = np.array(data["Y_force_template"], dtype=float)
                    out_f_add = f_t_all * (np.exp(out_f_l) - 1.0)
                else:
                    out_f_add = out_f_l
                # Add baseline first, then apply FORCE_PREDICT_MASK on absolute deltas
                dF_abs_l = out_f_add + np.array(data["Y_force_baseline"])
                if "FORCE_PREDICT_MASK" in data:
                    _mask = np.array(data["FORCE_PREDICT_MASK"], dtype=float).reshape(1, -1)
                    dF_abs_l = dF_abs_l * _mask
                # Apply VIA_PREDICT_MASK to residuals
                # Note: Masked residuals become zero, resulting in height-scaled via points:
                # P_final = P_template + P_baseline + 0 = P_template * h_ratio (pure height scaling)
                if APPLY_VIA_PREDICT_MASK and "FORCE_PREDICT_MASK" in data:
                    force_mask = data["FORCE_PREDICT_MASK"]
                    via_mask_list = []
                    for mi, m in enumerate(muscles):
                        L = int(via_len[m])
                        mask_val = float(force_mask[mi])
                        via_mask_list.extend([mask_val] * (3 * L))
                    # Apply extra mask if enabled
                    if APPLY_VIA_PREDICT_MASK_EXTRA and "VIA_PREDICT_MASK_EXTRA" in data:
                        via_mask_extra = data["VIA_PREDICT_MASK_EXTRA"]
                        via_mask_combined = []
                        for mi, m in enumerate(muscles):
                            L = int(via_len[m])
                            combined_mask = float(force_mask[mi]) * float(via_mask_extra[mi])
                            via_mask_combined.extend([combined_mask] * (3 * L))
                        via_mask_list = via_mask_combined
                    _via_mask = np.array(via_mask_list, dtype=float).reshape(1, -1)
                    out_v_l = out_v_l * _via_mask
                dF_pred_lat = dF_abs_l
                dVia_pred_lat = (out_v_l + np.array(data["Y_via_baseline"]))
        except Exception as e:
            print("Warning: Latent model inference failed; skipping latent revised outputs:", e)

    # GNN predictions (optional)
    dF_pred_gnn = dVia_pred_gnn = None
    if ("gnn" in models_to_use) and ((MODELS_DIR / "gnn.pt").exists()) and ("X_seq" in data) and ("X_seq_mask" in data):
        try:
            Yf_res_g, Yv_res_g = predict_gnn(MODELS_DIR / "gnn.pt", data["X_seq"], data["X_seq_mask"], stats, muscles, via_len)
            # Apply FORCE_PREDICT_MASK to residuals
            if "FORCE_PREDICT_MASK" in data:
                _mask = np.array(data["FORCE_PREDICT_MASK"], dtype=float).reshape(1, -1)
                Yf_res_g = np.array(Yf_res_g, dtype=float) * _mask
            # Apply VIA_PREDICT_MASK to residuals
            if APPLY_VIA_PREDICT_MASK and "FORCE_PREDICT_MASK" in data:
                force_mask = data["FORCE_PREDICT_MASK"]
                via_mask_list = []
                for mi, m in enumerate(muscles):
                    L = int(via_len[m])
                    mask_val = float(force_mask[mi])
                    via_mask_list.extend([mask_val] * (3 * L))
                # Apply extra mask if enabled
                if APPLY_VIA_PREDICT_MASK_EXTRA and "VIA_PREDICT_MASK_EXTRA" in data:
                    via_mask_extra = data["VIA_PREDICT_MASK_EXTRA"]
                    via_mask_combined = []
                    for mi, m in enumerate(muscles):
                        L = int(via_len[m])
                        combined_mask = float(force_mask[mi]) * float(via_mask_extra[mi])
                        via_mask_combined.extend([combined_mask] * (3 * L))
                    via_mask_list = via_mask_combined
                _via_mask = np.array(via_mask_list, dtype=float).reshape(1, -1)
                Yv_res_g = np.array(Yv_res_g, dtype=float) * _via_mask
            dF_pred_gnn = (np.array(Yf_res_g) + np.array(data["Y_force_baseline"]))
            dVia_pred_gnn = (np.array(Yv_res_g) + np.array(data["Y_via_baseline"]))
        except Exception as e:
            print("Warning: GNN inference failed; skipping gnn revised outputs:", e)

    # Pre-load AUX data for age extraction (used in age-adjusted baseline)
    aux_data_for_age = data.get("AUX", [])

    # Write both baseline and revised models (revised = template + predicted deltas)
    for i, osim_path in enumerate(IDS):
        # Determine template by sex
        sex_flag = 1.0 if "/Male/" in osim_path else 0.0
        data_root = require_data_root()
        tmpl_path = male_template_path(data_root) if sex_flag > 0.5 else female_template_path(data_root)
        base_model = OSIMModel.from_file(tmpl_path)
        viabase_model = None
        # For MLP predictions: pick sex-specific or neutral checkpoints row-wise
        dF_pred_row = None; dVia_pred_row = None
        if "mlp" in models_to_use:
            print("Running MLP prediction")  # Debug
            try:
                mu = stats.get("X", {}).get("mean")
                sd = stats.get("X", {}).get("std")
                x_row = X[i]
                Xn_row = [(v - m) / s for v, m, s in zip(x_row, mu, sd)] if (mu is not None and sd is not None) else x_row

                # Append baseline summary features (bf,bv) if enabled (must match training setting)
                if USE_BASELINE_SUMMARY_FEATURES:
                    dF_bl_row = data["Y_force_baseline"][i]
                    dVia_bl_row = data["Y_via_baseline"][i]
                    bf = float(np.mean(np.abs(np.array(dF_bl_row, dtype=float))))
                    bv = float(np.mean(np.abs(np.array(dVia_bl_row, dtype=float))))
                    Xn_row_aug = list(Xn_row) + [bf, bv]
                else:
                    Xn_row_aug = list(Xn_row)

                if USE_SEX_SPECIFIC and sex_flag > 0.5 and (MODELS_DIR / "mlp_force_m.pt").exists():
                    ckpt_force = load_checkpoint(MODELS_DIR / "mlp_force_m.pt")
                    ckpt_via = load_checkpoint(MODELS_DIR / "mlp_via_m.pt")
                elif USE_SEX_SPECIFIC and sex_flag <= 0.5 and (MODELS_DIR / "mlp_force_f.pt").exists():
                    ckpt_force = load_checkpoint(MODELS_DIR / "mlp_force_f.pt")
                    ckpt_via = load_checkpoint(MODELS_DIR / "mlp_via_f.pt")
                else:
                    ckpt_force = load_checkpoint(MODELS_DIR / "mlp_force.pt")
                    ckpt_via = load_checkpoint(MODELS_DIR / "mlp_via.pt")
                dF_pred_row = predict(ckpt_force, [Xn_row_aug])[0]
                # Apply FORCE_PREDICT_MASK to residuals for this row
                try:
                    _mask = np.array(data.get("FORCE_PREDICT_MASK") or [1]*len(muscles), dtype=float)
                    dF_pred_row = (np.array(dF_pred_row, dtype=float) * _mask).tolist()
                except Exception:
                    pass
                dVia_pred_row = predict(ckpt_via, [Xn_row_aug])[0]
                print(f"Prediction successful for {osim_path}: dF_pred_row[0] = {dF_pred_row[0] if dF_pred_row else None}")  # Debug
            except Exception as e:
                print(f"Prediction failed for {osim_path}: {e}")  # Debug
                dF_pred_row = None
                dVia_pred_row = None

        # Random Forest predictions
        dF_pred_rf = None; dVia_pred_rf = None
        if "rf" in models_to_use:
            print("Running Random Forest prediction")
            try:
                mu = stats.get("X", {}).get("mean")
                sd = stats.get("X", {}).get("std")
                x_row = X[i]
                Xn_row = [(v - m) / s for v, m, s in zip(x_row, mu, sd)] if (mu is not None and sd is not None) else x_row

                # Append baseline summary features (bf,bv) if enabled (must match training setting)
                if USE_BASELINE_SUMMARY_FEATURES:
                    dF_bl_row = data["Y_force_baseline"][i]
                    dVia_bl_row = data["Y_via_baseline"][i]
                    bf = float(np.mean(np.abs(np.array(dF_bl_row, dtype=float))))
                    bv = float(np.mean(np.abs(np.array(dVia_bl_row, dtype=float))))
                    Xn_row_aug = list(Xn_row) + [bf, bv]
                else:
                    Xn_row_aug = list(Xn_row)

                # Random Forest models don't have sex-specific versions yet
                rf_force_path = MODELS_DIR / "rf_force.pkl"
                rf_via_path = MODELS_DIR / "rf_via.pkl"

                if rf_force_path.exists() and rf_via_path.exists():
                    dF_pred_rf = predict_random_forest(rf_force_path, [Xn_row_aug])[0]
                    # Apply FORCE_PREDICT_MASK
                    try:
                        _mask = np.array(data.get("FORCE_PREDICT_MASK") or [1]*len(muscles), dtype=float)
                        dF_pred_rf = (np.array(dF_pred_rf, dtype=float) * _mask).tolist()
                    except Exception:
                        pass
                    dVia_pred_rf = predict_random_forest(rf_via_path, [Xn_row_aug])[0]

                    # Apply VIA_PREDICT_MASK to RF via predictions
                    if APPLY_VIA_PREDICT_MASK and "FORCE_PREDICT_MASK" in data:
                        force_mask = data["FORCE_PREDICT_MASK"]
                        via_mask_list = []
                        for mi, m in enumerate(muscles):
                            L = int(via_len[m])
                            mask_val = float(force_mask[mi])
                            via_mask_list.extend([mask_val] * (3 * L))
                        # Apply extra mask if enabled
                        if APPLY_VIA_PREDICT_MASK_EXTRA and "VIA_PREDICT_MASK_EXTRA" in data:
                            via_mask_extra = data["VIA_PREDICT_MASK_EXTRA"]
                            via_mask_combined = []
                            for mi, m in enumerate(muscles):
                                L = int(via_len[m])
                                combined_mask = float(force_mask[mi]) * float(via_mask_extra[mi])
                                via_mask_combined.extend([combined_mask] * (3 * L))
                            via_mask_list = via_mask_combined
                        _via_mask = np.array(via_mask_list, dtype=float)
                        dVia_pred_rf = (np.array(dVia_pred_rf, dtype=float) * _via_mask).tolist()

                    # De-standardize predictions (RF outputs are in standardized space)
                    if FAIR:
                        force_stats_key = "Y_force_log" if (FORCE_LOG and ("Y_force_log" in stats)) else "Y_force_res"
                        yf_mu = np.array(stats[force_stats_key]["mean"], dtype=float)
                        yf_sd = np.array(stats[force_stats_key]["std"], dtype=float)
                        yv_mu = np.array(stats["Y_via_res"]["mean"], dtype=float)
                        yv_sd = np.array(stats["Y_via_res"]["std"], dtype=float)
                        dF_pred_rf = ((np.array(dF_pred_rf) * yf_sd) + yf_mu).tolist()
                        dVia_pred_rf = ((np.array(dVia_pred_rf) * yv_sd) + yv_mu).tolist()

                    print(f"RF prediction successful for {osim_path}")
                else:
                    print(f"RF models not found at {rf_force_path}")
            except Exception as e:
                print(f"RF prediction failed for {osim_path}: {e}")
                dF_pred_rf = None
                dVia_pred_rf = None

        rev_model = OSIMModel.from_file(tmpl_path) if (dF_pred_row is not None and dVia_pred_row is not None) else None
        rev_model_transf = OSIMModel.from_file(tmpl_path) if (dF_pred_transf is not None and dVia_pred_transf is not None) else None
        rev_model_latent = OSIMModel.from_file(tmpl_path) if (dF_pred_lat is not None and dVia_pred_lat is not None) else None
        rev_model_rf = OSIMModel.from_file(tmpl_path) if (dF_pred_rf is not None and dVia_pred_rf is not None) else None
        age_adj_model = OSIMModel.from_file(tmpl_path) if (ENABLE_AGE_ADJUSTED_BASELINE and age_force_data is not None) else None

        # Baseline deltas from dataset (persisted by build_dataset.py)
        dF_bl = data["Y_force_baseline"][i]
        dVia_bl = data["Y_via_baseline"][i]

        # Apply baseline to base_model (note: Y_via_baseline is now in mm; convert to meters) if enabled
        if ENABLE_BASELINE:
            viabase_model = OSIMModel.from_file(tmpl_path)
            # Force
            for mi, m in enumerate(muscles):
                mt = base_model.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                # If FORCE_PREDICT_MASK excludes this muscle, keep baseline delta at 0
                if ("FORCE_PREDICT_MASK" in data) and (int(data["FORCE_PREDICT_MASK"][mi]) == 0):
                    delta = 0.0
                else:
                    delta = float(dF_bl[mi])
                base_model.set_muscle_max_isometric_force(m, float(f_t + delta))
            # Via
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = dVia_bl[off:off+3]
                    off += 3
                    pt = (base_model.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    # Convert via deltas to meters based on dataset flag
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    base_model.set_muscle_path_point_location(m, k, new_loc)
                    if viabase_model is not None:
                        viabase_model.set_muscle_path_point_location(m, k, new_loc)

        # Baseline comment: include weight and height ratios (compute against UNMODIFIED template)
        # Use a fresh template instance for ratio computation to avoid baseline edits affecting ratios
        subj = OSIMModel.from_file(osim_path)
        tmpl_for_ratio = OSIMModel.from_file(tmpl_path)
        w_ratio = _sacrum_mass_ratio(subj, tmpl_for_ratio)
        # Height ratio via CORB x-ratio (fallback to legacy)
        _hr = _corb_height_ratio_via_x(subj, tmpl_for_ratio)
        if _hr is None:
            h_subj = _height_estimate(subj)
            h_tmpl = _height_estimate(tmpl_for_ratio)
            h_ratio = (h_subj / h_tmpl) if h_tmpl else 1.0
        else:
            h_ratio = float(_hr)
        comment = ET.Comment(f"Baseline ratios: weight_ratio={w_ratio:.6f}, height_ratio={h_ratio:.6f}, source={osim_path}")
        base_model.root.insert(0, comment)
        if viabase_model is not None:
            viabase_comment = ET.Comment(
                f"Viabaseline ratios: weight_ratio={w_ratio:.6f}, height_ratio={h_ratio:.6f}, source={osim_path}, note=forces kept generic"
            )
            viabase_model.root.insert(0, viabase_comment)

        # Scale body masses for all models by weight ratio
        scale_body_masses(base_model, w_ratio)
        if viabase_model is not None:
            scale_body_masses(viabase_model, w_ratio)
        if rev_model is not None:
            scale_body_masses(rev_model, w_ratio)
        if rev_model_transf is not None:
            scale_body_masses(rev_model_transf, w_ratio)
        if rev_model_latent is not None:
            scale_body_masses(rev_model_latent, w_ratio)
        if age_adj_model is not None:
            scale_body_masses(age_adj_model, w_ratio)

        # Apply model predictions to rev_model
        # Force (note: all force paths expect additive residuals; our models now may output log; writers get absolute already)
        if rev_model is not None:
            # Insert comment about model provenance and settings
            rev_comment = ET.Comment(f"Revised by: model=MLP, sex_specific={'true' if USE_SEX_SPECIFIC else 'false'}, source_ckpt={'sexed' if USE_SEX_SPECIFIC else 'neutral'}")
            rev_model.root.insert(0, rev_comment)
            # Apply mask to predicted residual first
            print(f"Checking mask for {osim_path}")  # Debug
            if "FORCE_PREDICT_MASK" in data:
                _mask = np.array(data["FORCE_PREDICT_MASK"], dtype=float)
                print(f"Mask for CORB: {_mask[0]}")  # Debug
                dF_pred_row = np.array(dF_pred_row, dtype=float) * _mask
            else:
                print("No FORCE_PREDICT_MASK in data")  # Debug
            # Start from template but add baseline delta + predicted residual to match absolute target
            for mi, m in enumerate(muscles):
                mt = rev_model.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                abs_delta = float(dF_bl[mi]) + float(dF_pred_row[mi])
                # Debug print for CORB
                if m == "CORB":
                    print(f"CORB: f_t={f_t}, dF_bl={dF_bl[mi]}, dF_pred_row={dF_pred_row[mi]}, abs_delta={abs_delta}")
                rev_model.set_muscle_max_isometric_force(m, float(f_t + abs_delta))
            # Via (predicted residuals might be in mm; convert based on dataset flag)
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    # Add baseline delta and residual delta
                    bdx, bdy, bdz = map(float, dVia_bl[off:off+3])
                    rdx, rdy, rdz = map(float, dVia_pred_row[off:off+3])
                    off += 3
                    pt = (rev_model.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    # Convert via deltas to meters based on dataset flag
                    new_loc = [loc0[0] + via_to_meters(bdx + rdx), loc0[1] + via_to_meters(bdy + rdy), loc0[2] + via_to_meters(bdz + rdz)]
                    rev_model.set_muscle_path_point_location(m, k, new_loc)

        # Apply transformer predictions to rev_model_transf
        if rev_model_transf is not None:
            revt_comment = ET.Comment(f"Revised by: model=Transformer, sex_specific={'true' if USE_SEX_SPECIFIC else 'false'}")
            rev_model_transf.root.insert(0, revt_comment)
            for mi, m in enumerate(muscles):
                mt = rev_model_transf.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                delta_abs = float(dF_pred_transf[i, mi])
                if ("FORCE_PREDICT_MASK" in data) and (int(data["FORCE_PREDICT_MASK"][mi]) == 0):
                    delta_abs = 0.0
                rev_model_transf.set_muscle_max_isometric_force(m, float(f_t + delta_abs))
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = map(float, dVia_pred_transf[i, off:off+3])
                    off += 3
                    pt = (rev_model_transf.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    rev_model_transf.set_muscle_path_point_location(m, k, new_loc)

        # Apply latent predictions
        if rev_model_latent is not None:
            revl_comment = ET.Comment(f"Revised by: model=Latent, sex_specific={'true' if USE_SEX_SPECIFIC else 'false'}")
            rev_model_latent.root.insert(0, revl_comment)
            for mi, m in enumerate(muscles):
                mt = rev_model_latent.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                delta_abs = float(dF_pred_lat[i, mi])
                if ("FORCE_PREDICT_MASK" in data) and (int(data["FORCE_PREDICT_MASK"][mi]) == 0):
                    delta_abs = 0.0
                rev_model_latent.set_muscle_max_isometric_force(m, float(f_t + delta_abs))
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = map(float, dVia_pred_lat[i, off:off+3])
                    off += 3
                    pt = (rev_model_latent.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    rev_model_latent.set_muscle_path_point_location(m, k, new_loc)

        # Apply Random Forest predictions
        if rev_model_rf is not None:
            rf_comment = ET.Comment(f"Revised by: model=RandomForest, sex_specific={'true' if USE_SEX_SPECIFIC else 'false'}")
            rev_model_rf.root.insert(0, rf_comment)
            for mi, m in enumerate(muscles):
                mt = rev_model_rf.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                delta_abs = float(dF_pred_rf[mi])
                if ("FORCE_PREDICT_MASK" in data) and (int(data["FORCE_PREDICT_MASK"][mi]) == 0):
                    delta_abs = 0.0
                rev_model_rf.set_muscle_max_isometric_force(m, float(f_t + delta_abs))
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = float(dVia_pred_rf[off]), float(dVia_pred_rf[off+1]), float(dVia_pred_rf[off+2])
                    off += 3
                    pt = (rev_model_rf.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    rev_model_rf.set_muscle_path_point_location(m, k, new_loc)

        # Apply age-adjusted baseline (template with age-based force adjustments)
        if age_adj_model is not None:
            # Get subject age from pre-loaded AUX data
            if i < len(aux_data_for_age):
                subj_age = float(aux_data_for_age[i].get("age", 65.0))
            else:
                subj_age = 65.0  # reference age

            sex_char = "M" if sex_flag > 0.5 else "F"
            age_adj_comment = ET.Comment(
                f"Age-adjusted baseline: age={subj_age:.1f}, sex={sex_char}, reference_age=65.0, height_scaled_via_points=True, source={osim_path}"
            )
            age_adj_model.root.insert(0, age_adj_comment)

            # Adjust max isometric forces based on age
            for mi, m in enumerate(muscles):
                mt = age_adj_model.data["forces"].get(m) or {}
                f_template = mt.get("max_isometric_force") or 0.0
                f_adjusted = compute_age_adjusted_force(f_template, subj_age, sex_char, m, age_force_data)
                age_adj_model.set_muscle_max_isometric_force(m, float(f_adjusted))

            # Scale via points (muscle positions) based on height ratio (same as baseline)
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = dVia_bl[off:off+3]
                    off += 3
                    pt = (age_adj_model.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    # Convert via deltas to meters based on dataset flag
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    age_adj_model.set_muscle_path_point_location(m, k, new_loc)

        # Determine sex for template properties lookup
        sex_str = "male" if sex_flag > 0.5 else "female"

        # Scale muscle fiber properties based on muscle length changes
        if ENABLE_BASELINE and template_props:
            scale_muscle_fiber_properties(base_model, muscles, template_props, sex_str)
            if viabase_model is not None:
                scale_muscle_fiber_properties(viabase_model, muscles, template_props, sex_str)
        if rev_model is not None and template_props:
            scale_muscle_fiber_properties(rev_model, muscles, template_props, sex_str)
        if rev_model_transf is not None and template_props:
            scale_muscle_fiber_properties(rev_model_transf, muscles, template_props, sex_str)
        if rev_model_latent is not None and template_props:
            scale_muscle_fiber_properties(rev_model_latent, muscles, template_props, sex_str)
        if rev_model_rf is not None and template_props:
            scale_muscle_fiber_properties(rev_model_rf, muscles, template_props, sex_str)
        if age_adj_model is not None and template_props:
            scale_muscle_fiber_properties(age_adj_model, muscles, template_props, sex_str)

        base_out = OUT_DIR / (Path(osim_path).stem + "_baseline.osim")
        viabase_out = OUT_DIR / (Path(osim_path).stem + "_viabaseline.osim")
        rev_out = OUT_DIR / (Path(osim_path).stem + "_revised.osim")
        rev_out_transf = OUT_DIR / (Path(osim_path).stem + "_revised_transf.osim")
        rev_out_latent = OUT_DIR / (Path(osim_path).stem + "_revised_latent.osim")
        rev_out_rf = OUT_DIR / (Path(osim_path).stem + "_revised_rf.osim")
        age_adj_out = OUT_DIR / (Path(osim_path).stem + "_age_adjusted.osim")
        if ENABLE_BASELINE:
            base_model.save(base_out)
            if viabase_model is not None:
                viabase_model.save(viabase_out)
        if rev_model is not None:
            rev_model.save(rev_out)
        if rev_model_transf is not None:
            rev_model_transf.save(rev_out_transf)
        if rev_model_latent is not None:
            rev_model_latent.save(rev_out_latent)
        if rev_model_rf is not None:
            rev_model_rf.save(rev_out_rf)
        if age_adj_model is not None:
            age_adj_model.save(age_adj_out)
        if dF_pred_gnn is not None and dVia_pred_gnn is not None and ("gnn" in models_to_use):
            rev_model_gnn = OSIMModel.from_file(tmpl_path)
            gnn_comment = ET.Comment(f"Revised by: model=GNN, sex_specific={'true' if USE_SEX_SPECIFIC else 'false'}")
            rev_model_gnn.root.insert(0, gnn_comment)
            # Scale body masses
            scale_body_masses(rev_model_gnn, w_ratio)
            for mi, m in enumerate(muscles):
                mt = rev_model_gnn.data["forces"].get(m) or {}
                f_t = mt.get("max_isometric_force") or 0.0
                delta_abs = float(dF_pred_gnn[i, mi])
                if ("FORCE_PREDICT_MASK" in data) and (int(data["FORCE_PREDICT_MASK"][mi]) == 0):
                    delta_abs = 0.0
                rev_model_gnn.set_muscle_max_isometric_force(m, float(f_t + delta_abs))
            off = 0
            for m in muscles:
                L = via_len[m]
                for k in range(L):
                    dx, dy, dz = map(float, dVia_pred_gnn[i, off:off+3])
                    off += 3
                    pt = (rev_model_gnn.data["forces"][m].get("path_points") or [])[k]
                    loc0 = pt.get("location") or [0.0, 0.0, 0.0]
                    new_loc = [loc0[0] + via_to_meters(dx), loc0[1] + via_to_meters(dy), loc0[2] + via_to_meters(dz)]
                    rev_model_gnn.set_muscle_path_point_location(m, k, new_loc)
            # Scale fiber properties for GNN model
            if template_props:
                scale_muscle_fiber_properties(rev_model_gnn, muscles, template_props, sex_str)
            rev_out_gnn = OUT_DIR / (Path(osim_path).stem + "_revised_gnn.osim")
            rev_model_gnn.save(rev_out_gnn)
        if ENABLE_BASELINE:
            print("Wrote:", base_out)
            if viabase_model is not None:
                print("Wrote:", viabase_out)
        if rev_model is not None:
            print("Wrote:", rev_out)
        if rev_model_transf is not None:
            print("Wrote:", rev_out_transf)
        if rev_model_latent is not None:
            print("Wrote:", rev_out_latent)
        if rev_model_rf is not None:
            print("Wrote:", rev_out_rf)
        if age_adj_model is not None:
            print("Wrote:", age_adj_out)
        if dF_pred_gnn is not None and dVia_pred_gnn is not None and ("gnn" in models_to_use):
            print("Wrote:", rev_out_gnn)


if __name__ == "__main__":
    main()

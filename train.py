import os, numpy as np, pandas as pd, torch, torch.nn as nn
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Significance: Pearson r and p-value (better-than-chance test) ---
def pearsonr_with_p(y_true, y_pred):
    """
    Compute Pearson correlation (r) and two-tailed p-value between arrays.
    Falls back to p=nan if SciPy is unavailable.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Handle edge cases
    if y_true.size < 3 or np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
        return float('nan'), float('nan')
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    # Clip numerical drift
    r = max(min(r, 1.0), -1.0)
    # Try SciPy for exact p-value; otherwise fallback to nan
    p = float('nan')
    try:
        from scipy import stats
        n = y_true.size
        # Student's t with n-2 dof
        t = r * np.sqrt(max(0.0, (n - 2) / max(1e-12, 1.0 - r * r)))
        p = float(2.0 * stats.t.sf(np.abs(t), df=n - 2))
    except Exception:
        pass
    return r, p

BASE = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset"
NPZ  = os.path.join(BASE, "Embedding", "dataset_T20.npz")
META = os.path.join(BASE, "metadata.csv")

# ---- CLI args ----
parser = argparse.ArgumentParser(description="Train multimodal disagreement regressors")
parser.add_argument("--mode", choices=["lstm", "lstm_meta"], default="lstm",
                    help="Which training pipeline to run")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--meta", type=str, default="base",
                    help=("For --mode lstm_meta: which metadata to use. "
                          "Options: 'none' (no metadata), 'base' (Item,Gender,Duration), "
                          "'all' (Item,Gender,Duration,R,C,Consensus), or a comma list e.g. "
                          "'Item,Gender' or 'Item,Duration,Consensus'. Shorthand tokens allowed: "
                          "R/RChoice, C/CChoice."))
# Duration transformations for ablations
parser.add_argument("--log_duration", action="store_true",
                    help="Apply log1p(Duration) before scaling (effective when Duration is numeric, not binned)")
parser.add_argument("--duration_bins", type=int, default=0,
                    help="If >0, discretize Duration into that many quantile bins and one-hot encode; overrides numeric Duration")
args = parser.parse_args()

MODE   = args.mode
EPOCHS = args.epochs
BATCH  = args.batch
LR     = args.lr

def parse_meta_spec(spec: str, available_cols):
    if spec is None:
        spec = "base"
    s = spec.strip().lower()
    # Map user tokens to actual column names in metadata.csv
    TOK2COL = {
        "item": "Item", "gender": "Gender", "duration": "Duration",
        "r": "R Choice", "rchoice": "R Choice",
        "c": "C Choice", "cchoice": "C Choice",
        "consensus": "Consensus",
    }
    if s in ("none", "no", "off"):
        selected = []
    elif s in ("base", "basic"):
        selected = ["Item", "Gender", "Duration"]
    elif s in ("all", "full"):
        selected = ["Item", "Gender", "Duration", "R Choice", "C Choice", "Consensus"]
    else:
        toks = [t.strip() for t in s.split(",") if t.strip()]
        selected = [TOK2COL.get(t.lower(), t) for t in toks]
    # keep only columns that actually exist
    selected = [c for c in selected if c in available_cols]
    # split into types
    cat_cols = [c for c in ["Item", "Gender"] if c in selected]
    bin_cols = [c for c in ["R Choice", "C Choice", "Consensus"] if c in selected]
    num_cols = [c for c in ["Duration"] if c in selected]
    return selected, cat_cols, bin_cols, num_cols

OUT_DIR = os.path.join(BASE, "Results", MODE)
# If lstm_meta, append meta spec and duration transforms to subdir name for clearer ablation bookkeeping
if MODE == "lstm_meta":
    meta_tag = getattr(args, "meta", "base").replace(",", "_").replace(" ", "") or "custom"
    if getattr(args, "duration_bins", 0) and int(getattr(args, "duration_bins", 0)) > 0:
        meta_tag += f"_bins{int(args.duration_bins)}"
    elif getattr(args, "log_duration", False):
        meta_tag += "_log"
    OUT_DIR = os.path.join(OUT_DIR, meta_tag)
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if MODE == "lstm_meta":
    print(f"[RUN] mode={MODE} meta={args.meta} log_duration={args.log_duration} duration_bins={args.duration_bins} "
          f"epochs={EPOCHS} batch={BATCH} lr={LR} -> {OUT_DIR}")
else:
    print(f"[RUN] mode={MODE} epochs={EPOCHS} batch={BATCH} lr={LR} -> {OUT_DIR}")

# 1) load data
data  = np.load(NPZ, allow_pickle=True)
X, y, clips = data["X"].astype("float32"), data["y"].astype("float32"), data["clips"]
meta = pd.read_csv(META, encoding="latin-1")
key  = "File name" if "File name" in meta.columns else "File_name"
folds = meta.set_index(key).loc[clips, "Fold"].values
meta_df = meta.set_index(key).loc[clips]

# Infer true lengths per sequence from non-zero rows (padding rows are all-zeros)
nonzero_mask = (np.abs(X).sum(axis=2) > 0)
L = nonzero_mask.sum(axis=1).astype("int64")  # shape (N,)

# 2) model
class LSTMReg(nn.Module):
    def __init__(self, d_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim = hidden * (2 if bidir else 1)
        self.fc  = nn.Linear(out_dim, 1)
    def forward(self, x, lens):
        # x: (B, T, D); lens: (B,)
        x = self.norm(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        # h shape: (layers * num_directions, B, hidden)
        if self.rnn.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)  # concat last layer's fwd & bwd
        else:
            h_last = h[-1]
        return self.fc(h_last).squeeze(-1)

def run_fold(tr_idx, te_idx):
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])

    tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr), batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(TensorDataset(Xte, yte, Lte), batch_size=64, shuffle=False)

    model = LSTMReg(d_in=X.shape[2], hidden=256, layers=2, bidir=True).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()

    best, wait, patience = float('inf'), 0, 5
    preds, trues = None, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, lb in tr_dl:
            xb, yb, lb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE)
            pred = model(xb, lb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # quick validation each epoch (early stopping)
        model.eval()
        with torch.no_grad():
            fold_preds, fold_trues = [], []
            for xb, yb, lb in te_dl:
                p = model(xb.to(DEVICE), lb.to(DEVICE)).cpu().numpy()
                fold_preds.append(p); fold_trues.append(yb.numpy())
            preds = np.concatenate(fold_preds); trues = np.concatenate(fold_trues)
            mse = mean_squared_error(trues, preds)
            if mse + 1e-6 < best:
                best, wait = mse, 0
            else:
                wait += 1
                if wait >= patience:
                    break

    # Pearson correlation & p-value for this fold
    r_fold, p_fold = pearsonr_with_p(trues, preds)
    # metrics for this fold
    return {
        'MSE': mean_squared_error(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'R2' : r2_score(trues, preds),
        'preds': preds,
        'trues': trues,
        'r': r_fold,
        'p': p_fold,
    }

# --- Helpers for metadata (per-fold fit/transform) ---
def make_meta(tr_idx, te_idx, cat_cols=None, bin_cols=None, num_cols=None,
              log_duration=False, duration_bins=0):
    # default to all available if not provided (backward compatible)
    if cat_cols is None:
        cat_cols = [c for c in ["Item", "Gender"] if c in meta_df.columns]
    if bin_cols is None:
        bin_cols = [c for c in ["R Choice", "C Choice", "Consensus"] if c in meta_df.columns]
    if num_cols is None:
        num_cols = [c for c in ["Duration"] if c in meta_df.columns]

    df_tr = meta_df.iloc[tr_idx].copy()
    df_te = meta_df.iloc[te_idx].copy()

    cat_tr_parts, cat_te_parts = [], []
    # existing categorical (Item/Gender)
    if cat_cols:
        cat_tr_parts.append(df_tr[cat_cols])
        cat_te_parts.append(df_te[cat_cols])

    # optional: bin Duration (TRAIN-FIT ONLY) then one-hot as categorical, and remove numeric Duration
    if duration_bins and ("Duration" in num_cols):
        try:
            _, edges = pd.qcut(df_tr["Duration"], q=int(duration_bins), retbins=True, duplicates='drop')
        except ValueError:
            mn, mx = df_tr["Duration"].min(), df_tr["Duration"].max()
            edges = np.linspace(mn, mx, int(duration_bins)+1)
        edges = np.array(edges, dtype=float)
        if edges.size >= 2:
            eps = 1e-6
            edges[0]  = edges[0]  - eps
            edges[-1] = edges[-1] + eps
            df_tr["Duration_bin"] = pd.cut(df_tr["Duration"], bins=edges, include_lowest=True).astype(str)
            df_te["Duration_bin"] = pd.cut(df_te["Duration"], bins=edges, include_lowest=True).astype(str)
            cat_tr_parts.append(df_tr[["Duration_bin"]])
            cat_te_parts.append(df_te[["Duration_bin"]])
            num_cols = [c for c in num_cols if c != "Duration"]

    # OneHot encode all categorical pieces together
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    parts_tr, parts_te = [], []
    if cat_tr_parts:
        cat_tr = pd.concat(cat_tr_parts, axis=1)
        cat_te = pd.concat(cat_te_parts, axis=1)
        parts_tr.append(ohe.fit_transform(cat_tr))
        parts_te.append(ohe.transform(cat_te))

    # Binary Yes/No columns
    if bin_cols:
        parts_tr.append((df_tr[bin_cols] == "Yes").astype(int).values)
        parts_te.append((df_te[bin_cols] == "Yes").astype(int).values)

    # Numeric columns (Duration if not binned)
    if num_cols:
        Xtr_num = df_tr[num_cols].astype(float).copy()
        Xte_num = df_te[num_cols].astype(float).copy()
        if log_duration and ("Duration" in num_cols):
            Xtr_num["Duration"] = np.log1p(Xtr_num["Duration"])  # log(1+x)
            Xte_num["Duration"] = np.log1p(Xte_num["Duration"])
        scaler = StandardScaler()
        parts_tr.append(scaler.fit_transform(Xtr_num.values))
        parts_te.append(scaler.transform(Xte_num.values))

    Xtr_meta = np.concatenate(parts_tr, axis=1).astype("float32") if parts_tr else np.zeros((len(tr_idx), 0), dtype="float32")
    Xte_meta = np.concatenate(parts_te, axis=1).astype("float32") if parts_te else np.zeros((len(te_idx), 0), dtype="float32")
    return Xtr_meta, Xte_meta

def write_summary(out_dir, mode_tag, meta_tag, scores, overall_r=None, overall_p=None):
    """
    Save a standardized summary.csv under out_dir for later aggregation/plotting.
    scores: list of dicts like {'MSE':..., 'MAE':..., 'R2':..., 'r':..., 'p':...} per fold.
    """
    mse_mean = float(np.mean([s['MSE'] for s in scores]))
    mae_mean = float(np.mean([s['MAE'] for s in scores]))
    r2_mean  = float(np.mean([s['R2'] for s in scores]))
    rmse_mean = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    df = pd.DataFrame([{
        'Mode': mode_tag,
        'Meta': meta_tag or '',
        'MSE_mean': mse_mean,
        'MAE_mean': mae_mean,
        'R2_mean':  r2_mean,
        'RMSE_mean': rmse_mean,
        'r_mean': float(np.mean([s.get('r', np.nan) for s in scores])),
        'p_mean': float(np.nanmean([s.get('p', np.nan) for s in scores])),
        'r_overall': overall_r if overall_r is not None else np.nan,
        'p_overall': overall_p if overall_p is not None else np.nan,
    }])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return df.iloc[0].to_dict()

def _project_root():
    # BASE points to .../disagreement-dataset; project root is its parent
    return os.path.dirname(BASE)

def _fig_dir():
    return os.path.join(_project_root(), "Figures&Graphs")

def aggregate_all_summaries_and_plot():
    """Disabled plotting in train.py. Use the notebook ablation_report.ipynb to aggregate and plot."""
    print("[PLOT] Disabled in train.py. Please open ablation_report.ipynb to aggregate & generate figures.")

# --- Original LSTM CV (fixed Fold) ---
def run_cv_lstm():
    unique_folds = sorted(np.unique(folds))
    scores, all_rows = [], []
    for k in unique_folds:
        te_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]
        res = run_fold(tr_idx, te_idx)
        scores.append({'MSE': res['MSE'], 'MAE': res['MAE'], 'R2': res['R2'], 'r': res['r'], 'p': res['p']})
        rmse_k = float(np.sqrt(res['MSE']))
        print(f"Fold {k}: MSE={res['MSE']:.3f}, RMSE={rmse_k:.3f}, MAE={res['MAE']:.3f}, R2={res['R2']:.3f}, r={res['r']:.3f}, p={res['p']:.3g}")
        df_fold = pd.DataFrame({'clip': clips[te_idx], 'y_true': res['trues'], 'y_pred': res['preds'], 'fold': k})
        df_fold.to_csv(os.path.join(OUT_DIR, f"lstm_preds_fold{k}.csv"), index=False)
        all_rows.append(df_fold)
    mean_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    mean_rmse = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    print("Per-fold (fixed Fold):", scores)
    print("Mean:", {**mean_scores, 'RMSE': mean_rmse})
    pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "lstm_preds_all_folds.csv"), index=False)
    # Overall Pearson r/p across all test predictions (better-than-chance report)
    df_all = pd.concat(all_rows, ignore_index=True)
    r_all, p_all = pearsonr_with_p(df_all['y_true'].values, df_all['y_pred'].values)
    print(f"Overall across folds: r={r_all:.3f}, p={p_all:.3g}")
    print("Saved per-fold predictions to:", OUT_DIR)
    # Save standardized summary then aggregate & plot
    write_summary(OUT_DIR, mode_tag="lstm", meta_tag="", scores=scores, overall_r=r_all, overall_p=p_all)

# --- LSTM + metadata CV ---
class LSTMRegWithMeta(nn.Module):
    def __init__(self, d_in, m_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim + m_in, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, x, lens, xmeta):
        x = self.norm(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        if self.rnn.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h_last = h[-1]
        z = torch.cat([h_last, xmeta], dim=1)
        return self.head(z).squeeze(-1)

def run_fold_meta(tr_idx, te_idx, cat_cols, bin_cols, num_cols, log_duration=False, duration_bins=0):
    # sequence tensors
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])
    # metadata
    Mtr_np, Mte_np = make_meta(tr_idx, te_idx, cat_cols, bin_cols, num_cols,
                               log_duration=log_duration, duration_bins=duration_bins)
    Mtr, Mte = torch.tensor(Mtr_np), torch.tensor(Mte_np)

    tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr, Mtr), batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(TensorDataset(Xte, yte, Lte, Mte), batch_size=64, shuffle=False)

    model = LSTMRegWithMeta(d_in=X.shape[2], m_in=Mtr.shape[1], hidden=256, layers=2, bidir=True).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.MSELoss()

    best, wait, patience = float('inf'), 0, 5
    preds, trues = None, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb, lb, mb in tr_dl:
            xb, yb, lb, mb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE), mb.to(DEVICE)
            pred = model(xb, lb, mb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            fold_preds, fold_trues = [], []
            for xb, yb, lb, mb in te_dl:
                p = model(xb.to(DEVICE), lb.to(DEVICE), mb.to(DEVICE)).cpu().numpy()
                fold_preds.append(p); fold_trues.append(yb.numpy())
            preds = np.concatenate(fold_preds); trues = np.concatenate(fold_trues)
            mse = mean_squared_error(trues, preds)
            if mse + 1e-6 < best:
                best, wait = mse, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    r_fold, p_fold = pearsonr_with_p(trues, preds)
    return {
        'MSE': mean_squared_error(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'R2' : r2_score(trues, preds),
        'preds': preds,
        'trues': trues,
        'r': r_fold,
        'p': p_fold,
    }

def run_cv_lstm_meta():
    # Parse meta spec according to CLI for ablation
    selected, cat_cols, bin_cols, num_cols = parse_meta_spec(args.meta, meta_df.columns)
    print(f"[META] Using columns: {selected if selected else 'NONE'}")
    print(f"[META] log_duration={args.log_duration}, duration_bins={args.duration_bins}")

    unique_folds = sorted(np.unique(folds))
    scores, all_rows = [], []
    for k in unique_folds:
        te_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]
        res = run_fold_meta(tr_idx, te_idx, cat_cols, bin_cols, num_cols,
                            log_duration=args.log_duration, duration_bins=args.duration_bins)
        scores.append({'MSE': res['MSE'], 'MAE': res['MAE'], 'R2': res['R2'], 'r': res['r'], 'p': res['p']})
        rmse_k = float(np.sqrt(res['MSE']))
        print(f"[META] Fold {k}: MSE={res['MSE']:.3f}, RMSE={rmse_k:.3f}, MAE={res['MAE']:.3f}, R2={res['R2']:.3f}, r={res['r']:.3f}, p={res['p']:.3g}")
        df_fold = pd.DataFrame({'clip': clips[te_idx], 'y_true': res['trues'], 'y_pred': res['preds'], 'fold': k})
        df_fold.to_csv(os.path.join(OUT_DIR, f"lstm_meta_preds_fold{k}.csv"), index=False)
        all_rows.append(df_fold)
    mean_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    mean_rmse = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    print("[META] Per-fold:", scores)
    print("[META] Mean:", {**mean_scores, 'RMSE': mean_rmse})

    # Overall Pearson r/p across all test predictions (better-than-chance report)
    df_all = pd.concat(all_rows, ignore_index=True)
    r_all, p_all = pearsonr_with_p(df_all['y_true'].values, df_all['y_pred'].values)
    print(f"[META] Overall across folds: r={r_all:.3f}, p={p_all:.3g}")

    write_summary(OUT_DIR, mode_tag="lstm_meta", meta_tag=args.meta, scores=scores, overall_r=r_all, overall_p=p_all)
    print("Saved META per-fold predictions to:", OUT_DIR)


if __name__ == "__main__":
    if MODE == "lstm":
        run_cv_lstm()
    elif MODE == "lstm_meta":
        run_cv_lstm_meta()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
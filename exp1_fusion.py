

"""
Experiment 1 — Fusion Baselines (no metadata)
- Early fusion: BiLSTM over concatenated txt+aud sequences  (--mode lstm)
- Late fusion: Stacking (Ridge over clip-level txt/aud, then meta-level Ridge) (--mode stacking)
- Hybrid fusion: txt-branch BiLSTM + aud-branch BiLSTM -> concat sequence -> fusion BiLSTM (--mode hybrid)

This script writes ONLY CSVs (no plots):
- Results/exp1/<mode>/*_preds_foldK.csv
- Results/exp1/<mode>/<mode>_preds_all_folds.csv
- Results/exp1/<mode>/summary.csv   (MSE_mean, RMSE_mean, MAE_mean, R2_mean, r_mean, p_mean, r_overall, p_overall)

Default run executes all three modes sequentially: --mode all
"""

import os, numpy as np, pandas as pd, torch, torch.nn as nn
import argparse
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -------------------- Config --------------------
BASE = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset"
NPZ  = os.path.join(BASE, "Embedding", "dataset_T20.npz")
META = os.path.join(BASE, "metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_DIM = 768  # first 768 dims are text features; change if different

# -------------------- Utils --------------------
def pearsonr_with_p(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 3 or np.allclose(np.std(y_true), 0) or np.allclose(np.std(y_pred), 0):
        return float('nan'), float('nan')
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r = max(min(r, 1.0), -1.0)
    try:
        from scipy import stats
        n = y_true.size
        t = r * np.sqrt(max(0.0, (n - 2) / max(1e-12, 1.0 - r * r)))
        p = float(2.0 * stats.t.sf(abs(t), df=n - 2))
    except Exception:
        p = float('nan')
    return r, p

def write_summary(out_dir, mode_tag, scores, overall_r=None, overall_p=None):
    mse_mean = float(np.mean([s['MSE'] for s in scores]))
    mae_mean = float(np.mean([s['MAE'] for s in scores]))
    r2_mean  = float(np.mean([s['R2']  for s in scores]))
    rmse_mean = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    df = pd.DataFrame([{
        'Mode': mode_tag,
        'Meta': '',
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

# -------------------- Data --------------------
data  = np.load(NPZ, allow_pickle=True)
X, y, clips = data["X"].astype("float32"), data["y"].astype("float32"), data["clips"]
meta = pd.read_csv(META, encoding="latin-1")
key  = "File name" if "File name" in meta.columns else "File_name"
folds = meta.set_index(key).loc[clips, "Fold"].values

# infer lengths from non-zero rows
L = (np.abs(X).sum(axis=2) > 0).sum(axis=1).astype("int64")

# -------------------- Early Fusion (LSTM) --------------------
class LSTMReg(nn.Module):
    def __init__(self, d_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim = hidden * (2 if bidir else 1)
        self.fc  = nn.Linear(out_dim, 1)
    def forward(self, x, lens):
        x = self.norm(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        h_last = torch.cat([h[-2], h[-1]], dim=1) if self.rnn.bidirectional else h[-1]
        return self.fc(h_last).squeeze(-1)

# -------------------- Hybrid Fusion --------------------
class BiLSTMBlock(nn.Module):
    def __init__(self, d_in, hidden=128, layers=1, bidir=True, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidir, dropout=dropout if layers>1 else 0.0)
        self.bidir = bidir; self.hidden = hidden
    @property
    def out_dim(self):
        return self.hidden * (2 if self.bidir else 1)
    def forward(self, x, lens):
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        h_last = torch.cat([h[-2], h[-1]], dim=1) if self.bidir else h[-1]
        return out, h_last

class HybridReg(nn.Module):
    def __init__(self, d_txt, d_aud, hidden=256, layers=1, bidir=True, dropout=0.1):
        super().__init__()
        h_half = max(1, hidden // 2)
        self.txt_branch = BiLSTMBlock(d_txt, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        self.aud_branch = BiLSTMBlock(d_aud, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        self.fusion = BiLSTMBlock(self.txt_branch.out_dim + self.aud_branch.out_dim,
                                  hidden=hidden, layers=layers, bidir=bidir, dropout=dropout)
        self.head = nn.Linear(self.fusion.out_dim, 1)
        self.norm_txt = nn.LayerNorm(d_txt)
        self.norm_aud = nn.LayerNorm(d_aud)
    def forward(self, x, lens):
        x_txt = self.norm_txt(x[:, :, :TEXT_DIM])
        x_aud = self.norm_aud(x[:, :, TEXT_DIM:])
        seq_t, _ = self.txt_branch(x_txt, lens)
        seq_a, _ = self.aud_branch(x_aud, lens)
        fused_seq = torch.cat([seq_t, seq_a], dim=-1)
        _, h = self.fusion(fused_seq, lens)
        return self.head(h).squeeze(-1)

# -------------------- Late Fusion (Stacking) --------------------
def masked_mean(seq_np, lens_np):
    N, T, D = seq_np.shape
    mask = (np.arange(T)[None, :] < lens_np[:, None]).astype(np.float32)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return (seq_np * mask[:, :, None]).sum(axis=1) / denom

def run_fold_lstm(tr_idx, te_idx, epochs, batch, lr):
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])

    tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr), batch_size=batch, shuffle=True)
    te_dl = DataLoader(TensorDataset(Xte, yte, Lte), batch_size=64, shuffle=False)

    model = LSTMReg(d_in=X.shape[2], hidden=256, layers=2, bidir=True).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    best, wait, patience = float('inf'), 0, 5
    for epoch in range(epochs):
        model.train()
        for xb, yb, lb in tr_dl:
            xb, yb, lb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE)
            pred = model(xb, lb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # early stop by val MSE
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb, lb in te_dl:
                p = model(xb.to(DEVICE), lb.to(DEVICE)).cpu().numpy()
                preds.append(p); trues.append(yb.numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        mse = mean_squared_error(trues, preds)
        if mse + 1e-6 < best: best, wait = mse, 0
        else:
            wait += 1
            if wait >= patience: break
    r, p = pearsonr_with_p(trues, preds)
    return {
        'MSE': mean_squared_error(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'R2' : r2_score(trues, preds),
        'preds': preds, 'trues': trues, 'r': r, 'p': p
    }

def run_fold_hybrid(tr_idx, te_idx, epochs, batch, lr):
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])

    tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr), batch_size=batch, shuffle=True)
    te_dl = DataLoader(TensorDataset(Xte, yte, Lte), batch_size=64, shuffle=False)

    d_txt = TEXT_DIM
    d_aud = X.shape[2] - d_txt
    model = HybridReg(d_txt=d_txt, d_aud=d_aud, hidden=256, layers=1, bidir=True, dropout=0.1).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    best, wait, patience = float('inf'), 0, 5
    for epoch in range(epochs):
        model.train()
        for xb, yb, lb in tr_dl:
            xb, yb, lb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE)
            pred = model(xb, lb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb, lb in te_dl:
                p = model(xb.to(DEVICE), lb.to(DEVICE)).cpu().numpy()
                preds.append(p); trues.append(yb.numpy())
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        mse = mean_squared_error(trues, preds)
        if mse + 1e-6 < best: best, wait = mse, 0
        else:
            wait += 1
            if wait >= patience: break
    r, p = pearsonr_with_p(trues, preds)
    return {
        'MSE': mean_squared_error(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'R2' : r2_score(trues, preds),
        'preds': preds, 'trues': trues, 'r': r, 'p': p
    }

def run_fold_stacking(tr_idx, te_idx):
    # clip-level masked mean
    Xtr_clip = masked_mean(X[tr_idx], L[tr_idx])
    Xte_clip = masked_mean(X[te_idx], L[te_idx])
    ytr_np, yte_np = y[tr_idx], y[te_idx]

    d_txt = TEXT_DIM
    Xtr_txt, Xtr_aud = Xtr_clip[:, :d_txt], Xtr_clip[:, d_txt:]
    Xte_txt, Xte_aud = Xte_clip[:, :d_txt], Xte_clip[:, d_txt:]

    s_txt, s_aud = StandardScaler(), StandardScaler()
    Xtr_txt = s_txt.fit_transform(Xtr_txt); Xte_txt = s_txt.transform(Xte_txt)
    Xtr_aud = s_aud.fit_transform(Xtr_aud); Xte_aud = s_aud.transform(Xte_aud)

    base_txt = Ridge(alpha=1.0); base_aud = Ridge(alpha=1.0)
    base_txt.fit(Xtr_txt, ytr_np); base_aud.fit(Xtr_aud, ytr_np)
    p_tr = np.column_stack([base_txt.predict(Xtr_txt), base_aud.predict(Xtr_aud)])
    p_te = np.column_stack([base_txt.predict(Xte_txt), base_aud.predict(Xte_aud)])

    meta_reg = Ridge(alpha=1.0)
    meta_reg.fit(p_tr, ytr_np)
    preds = meta_reg.predict(p_te)
    trues = yte_np
    r, p = pearsonr_with_p(trues, preds)
    return {
        'MSE': mean_squared_error(trues, preds),
        'MAE': mean_absolute_error(trues, preds),
        'R2' : r2_score(trues, preds),
        'preds': preds, 'trues': trues, 'r': r, 'p': p
    }

# -------------------- CV Loops --------------------
def run_cv(mode, epochs, batch, lr, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    uniq = sorted(np.unique(folds))
    scores, all_rows = [], []
    for k in uniq:
        te_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]
        if mode == 'lstm':
            res = run_fold_lstm(tr_idx, te_idx, epochs, batch, lr)
        elif mode == 'hybrid':
            res = run_fold_hybrid(tr_idx, te_idx, epochs, batch, lr)
        elif mode == 'stacking':
            res = run_fold_stacking(tr_idx, te_idx)
        else:
            raise ValueError(mode)
        scores.append({k: res[k] for k in ['MSE','MAE','R2','r','p']})
        rmse_k = float(np.sqrt(res['MSE']))
        tag = mode.upper() if mode!='lstm' else 'LSTM'
        print(f"[{tag}] Fold {k}: MSE={res['MSE']:.3f}, RMSE={rmse_k:.3f}, MAE={res['MAE']:.3f}, R2={res['R2']:.3f}, r={res['r']:.3f}, p={res['p']:.3g}")
        df_fold = pd.DataFrame({'clip': clips[te_idx], 'y_true': res['trues'], 'y_pred': res['preds'], 'fold': k})
        df_fold.to_csv(os.path.join(out_dir, f"{mode}_preds_fold{k}.csv"), index=False)
        all_rows.append(df_fold)
    mean_scores = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    mean_rmse = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    print(f"[{mode}] Per-fold:", scores)
    print(f"[{mode}] Mean:", {**mean_scores, 'RMSE': mean_rmse})
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, f"{mode}_preds_all_folds.csv"), index=False)
    r_all, p_all = pearsonr_with_p(all_df['y_true'].values, all_df['y_pred'].values)
    print(f"[{mode}] Overall across folds: r={r_all:.3f}, p={p_all:.3g}")
    write_summary(out_dir, mode_tag=mode, scores=scores, overall_r=r_all, overall_p=p_all)

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 1: fusion baselines (no metadata)")
    ap.add_argument('--mode', choices=['all','lstm','stacking','hybrid'], default='all')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch',  type=int, default=64)
    ap.add_argument('--lr',     type=float, default=3e-4)
    args = ap.parse_args()

    ROOT = os.path.join(os.path.dirname(BASE), 'Results', 'exp1')
    if args.mode in ('all','lstm'):
        out = os.path.join(ROOT, 'lstm'); print(f"[RUN exp1] lstm -> {out}")
        run_cv('lstm', args.epochs, args.batch, args.lr, out)
    if args.mode in ('all','stacking'):
        out = os.path.join(ROOT, 'stacking'); print(f"[RUN exp1] stacking -> {out}")
        run_cv('stacking', args.epochs, args.batch, args.lr, out)
    if args.mode in ('all','hybrid'):
        out = os.path.join(ROOT, 'hybrid'); print(f"[RUN exp1] hybrid -> {out}")
        run_cv('hybrid', args.epochs, args.batch, args.lr, out)
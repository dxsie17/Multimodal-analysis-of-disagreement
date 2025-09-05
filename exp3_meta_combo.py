import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset"
NPZ  = os.path.join(BASE, "Embedding", "dataset_T20.npz")
META = os.path.join(BASE, "metadata.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_DIM = 768  # first 768 dims are text; rest are audio

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

class HybridWithMetaClip(nn.Module):
    """Hybrid backbone + clip-level meta: txt-branch BiLSTM + aud-branch BiLSTM -> concat seq -> fusion BiLSTM -> last -> concat meta -> MLP head"""
    def __init__(self, d_txt, d_aud, m_in, hidden=256, layers=1, bidir=True, dropout=0.1):
        super().__init__()
        h_half = max(1, hidden // 2)
        self.txt_branch = BiLSTMBlock(d_txt, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        self.aud_branch = BiLSTMBlock(d_aud, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        self.fusion = BiLSTMBlock(self.txt_branch.out_dim + self.aud_branch.out_dim,
                                  hidden=hidden, layers=layers, bidir=bidir, dropout=dropout)
        self.norm_txt = nn.LayerNorm(d_txt)
        self.norm_aud = nn.LayerNorm(d_aud)
        self.head = nn.Sequential(nn.Linear(self.fusion.out_dim + m_in, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    def forward(self, x, lens, m):
        x_txt = self.norm_txt(x[:, :, :TEXT_DIM])
        x_aud = self.norm_aud(x[:, :, TEXT_DIM:])
        seq_t, _ = self.txt_branch(x_txt, lens)
        seq_a, _ = self.aud_branch(x_aud, lens)
        fused_seq = torch.cat([seq_t, seq_a], dim=-1)
        _, h = self.fusion(fused_seq, lens)
        z = torch.cat([h, m], dim=1)
        return self.head(z).squeeze(-1)

class HybridWithMetaTurn(nn.Module):
    """Hybrid backbone + turn-level meta broadcast: concatenate meta to each fused timestep before fusion BiLSTM"""
    def __init__(self, d_txt, d_aud, m_in, hidden=256, layers=1, bidir=True, dropout=0.1):
        super().__init__()
        h_half = max(1, hidden // 2)
        self.txt_branch = BiLSTMBlock(d_txt, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        self.aud_branch = BiLSTMBlock(d_aud, hidden=h_half, layers=layers, bidir=bidir, dropout=dropout)
        fused_in = self.txt_branch.out_dim + self.aud_branch.out_dim + m_in
        self.fusion = BiLSTMBlock(fused_in, hidden=hidden, layers=layers, bidir=bidir, dropout=dropout)
        self.norm_txt = nn.LayerNorm(d_txt)
        self.norm_aud = nn.LayerNorm(d_aud)
        self.fc = nn.Linear(self.fusion.out_dim, 1)
    def forward(self, x, lens, m):
        x_txt = self.norm_txt(x[:, :, :TEXT_DIM])
        x_aud = self.norm_aud(x[:, :, TEXT_DIM:])
        seq_t, _ = self.txt_branch(x_txt, lens)
        seq_a, _ = self.aud_branch(x_aud, lens)
        fused_seq = torch.cat([seq_t, seq_a], dim=-1)
        mexp = m.unsqueeze(1).expand(fused_seq.size(0), fused_seq.size(1), m.size(1))
        fused_aug = torch.cat([fused_seq, mexp], dim=-1)
        _, h = self.fusion(fused_aug, lens)
        return self.fc(h).squeeze(-1)

def pearsonr_with_p(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 3 or np.allclose(np.std(y_true),0) or np.allclose(np.std(y_pred),0):
        return float('nan'), float('nan')
    r = float(np.corrcoef(y_true, y_pred)[0,1]); r = max(min(r,1.0),-1.0)
    try:
        from scipy import stats
        n=y_true.size; t=r*np.sqrt(max(0.0,(n-2)/max(1e-12,1-r*r))); p=float(2*stats.t.sf(abs(t),df=n-2))
    except Exception: p=float('nan')
    return r,p

# ---------- data ----------
data  = np.load(NPZ, allow_pickle=True)
X, y, clips = data["X"].astype("float32"), data["y"].astype("float32"), data["clips"]
meta = pd.read_csv(META, encoding="latin-1")
key  = "File name" if "File name" in meta.columns else "File_name"
folds = meta.set_index(key).loc[clips, "Fold"].values
meta_df = meta.set_index(key).loc[clips]
L = (np.abs(X).sum(axis=2) > 0).sum(axis=1).astype("int64")

def build_meta(tr_idx, te_idx, cols, log_duration=False):
    TOK2COL = {"item":"Item","gender":"Gender","duration":"Duration",
               "r":"R Choice","rchoice":"R Choice","c":"C Choice","cchoice":"C Choice","consensus":"Consensus"}
    selected=[TOK2COL.get(t.strip().lower(), t.strip()) for t in cols]
    selected=[c for c in selected if c in meta_df.columns]
    cat_cols=[c for c in ["Item","Gender"] if c in selected]
    bin_cols=[c for c in ["R Choice","C Choice","Consensus"] if c in selected]
    num_cols=[c for c in ["Duration"] if c in selected]

    tr = meta_df.iloc[tr_idx].copy(); te = meta_df.iloc[te_idx].copy()
    parts_tr, parts_te = [], []

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        parts_tr.append(ohe.fit_transform(tr[cat_cols])); parts_te.append(ohe.transform(te[cat_cols]))
    if bin_cols:
        parts_tr.append((tr[bin_cols]=="Yes").astype(int).values)
        parts_te.append((te[bin_cols]=="Yes").astype(int).values)
    if num_cols:
        a = tr[num_cols].astype(float).copy(); b = te[num_cols].astype(float).copy()
        if log_duration and "Duration" in num_cols:
            a["Duration"]=np.log1p(a["Duration"]); b["Duration"]=np.log1p(b["Duration"])
        scaler = StandardScaler(); parts_tr.append(scaler.fit_transform(a.values)); parts_te.append(scaler.transform(b.values))

    Mtr = np.concatenate(parts_tr, axis=1).astype("float32") if parts_tr else np.zeros((len(tr_idx),0),dtype="float32")
    Mte = np.concatenate(parts_te, axis=1).astype("float32") if parts_te else np.zeros((len(te_idx),0),dtype="float32")
    return Mtr, Mte, selected

class LSTMClip(nn.Module):
    def __init__(self, d_in, m_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm=nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim=hidden*(2 if bidir else 1)
        self.head=nn.Sequential(nn.Linear(out_dim+m_in,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
    def forward(self, x,lens,m):
        x=self.norm(x)
        pack=nn.utils.rnn.pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_) = self.rnn(pack)
        h_last=torch.cat([h[-2],h[-1]],dim=1) if self.rnn.bidirectional else h[-1]
        return self.head(torch.cat([h_last,m],dim=1)).squeeze(-1)

class LSTMTurn(nn.Module):
    def __init__(self, d_in, m_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm=nn.LayerNorm(d_in+m_in)
        self.rnn = nn.LSTM(d_in+m_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim=hidden*(2 if bidir else 1)
        self.fc = nn.Linear(out_dim,1)
    def forward(self, x,lens,m):
        mexp = m.unsqueeze(1).expand(x.size(0), x.size(1), m.size(1))
        xa = torch.cat([x,mexp], dim=2); xa=self.norm(xa)
        pack=nn.utils.rnn.pack_padded_sequence(xa,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_) = self.rnn(pack)
        h_last=torch.cat([h[-2],h[-1]],dim=1) if self.rnn.bidirectional else h[-1]
        return self.fc(h_last).squeeze(-1)

def run_fold(inject, tr_idx, te_idx, cols, log_duration=False, epochs=40, batch=64, lr=3e-4, backbone='hybrid'):
    import torch
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])
    Mtr_np, Mte_np, used = build_meta(tr_idx, te_idx, cols, log_duration=log_duration)
    Mtr, Mte = torch.tensor(Mtr_np), torch.tensor(Mte_np)

    if backbone == 'hybrid':
        d_txt = TEXT_DIM; d_aud = X.shape[2] - d_txt
        if inject == 'clip':
            model = HybridWithMetaClip(d_txt=d_txt, d_aud=d_aud, m_in=Mtr.shape[1]).to(DEVICE)
        elif inject == 'turn':
            model = HybridWithMetaTurn(d_txt=d_txt, d_aud=d_aud, m_in=Mtr.shape[1]).to(DEVICE)
        else:
            raise ValueError(inject)
    elif backbone == 'lstm':
        if inject == 'clip':
            model = LSTMClip(d_in=X.shape[2], m_in=Mtr.shape[1]).to(DEVICE)
        elif inject == 'turn':
            model = LSTMTurn(d_in=X.shape[2], m_in=Mtr.shape[1]).to(DEVICE)
        else:
            raise ValueError(inject)
    else:
        raise ValueError(backbone)

    tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr, Mtr), batch_size=batch, shuffle=True)
    te_dl = DataLoader(TensorDataset(Xte, yte, Lte, Mte), batch_size=64, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4); lossf = nn.MSELoss()
    best, wait, patience = float('inf'), 0, 5
    preds = trues = None
    for ep in range(epochs):
        model.train()
        for xb, yb, lb, mb in tr_dl:
            xb, yb, lb, mb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE), mb.to(DEVICE)
            pb = model(xb, lb, mb)
            loss = lossf(pb, yb); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); P, T = [], []
        with torch.no_grad():
            for xb, yb, lb, mb in te_dl:
                p = model(xb.to(DEVICE), lb.to(DEVICE), mb.to(DEVICE)).cpu().numpy()
                P.append(p); T.append(yb.numpy())
        preds, trues = np.concatenate(P), np.concatenate(T)
        mse = mean_squared_error(trues, preds)
        if mse + 1e-6 < best: best, wait = mse, 0
        else:
            wait += 1
            if wait >= patience: break
    from sklearn.metrics import mean_absolute_error
    r, p = pearsonr_with_p(trues, preds)
    return {'MSE': mean_squared_error(trues, preds), 'MAE': mean_absolute_error(trues, preds), 'R2': r2_score(trues, preds), 'r': r, 'p': p, 'preds': preds, 'trues': trues}

def run_cv(inject, combo, out_dir, log_duration=False, epochs=40, batch=64, lr=3e-4, backbone='hybrid'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    scores, all_rows = [], []
    for k in sorted(np.unique(folds)):
        te_idx = np.where(folds==k)[0]; tr_idx = np.where(folds!=k)[0]
        res = run_fold(inject, tr_idx, te_idx, combo, log_duration=log_duration, epochs=epochs, batch=batch, lr=lr, backbone=backbone)
        scores.append({m:res[m] for m in ['MSE','MAE','R2','r','p']})
        rmse_k=float(np.sqrt(res['MSE']))
        print(f"[{backbone}|{inject}|{'+'.join(combo)}] Fold {k}: MSE={res['MSE']:.3f}, RMSE={rmse_k:.3f}, MAE={res['MAE']:.3f}, R2={res['R2']:.3f}, r={res['r']:.3f}, p={res['p']:.3g}")
        dfk = pd.DataFrame({'clip':clips[te_idx],'y_true':res['trues'],'y_pred':res['preds'],'fold':k})
        dfk.to_csv(os.path.join(out_dir, f"preds_fold{k}.csv"), index=False); all_rows.append(dfk)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, "preds_all_folds.csv"), index=False)
    r_all, p_all = pearsonr_with_p(all_df['y_true'].values, all_df['y_pred'].values)
    print(f"[{backbone}|{inject}|{'+'.join(combo)}] Overall: r={r_all:.3f}, p={p_all:.3g}")
    mse_mean=float(np.mean([s['MSE'] for s in scores])); mae_mean=float(np.mean([s['MAE'] for s in scores]))
    r2_mean=float(np.mean([s['R2'] for s in scores])); rmse_mean=float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    mode_tag = f"{backbone}_meta_{inject}"
    df = pd.DataFrame([{'Mode': mode_tag, 'Meta': ','.join(combo),
                        'MSE_mean':mse_mean,'MAE_mean':mae_mean,'R2_mean':r2_mean,'RMSE_mean':rmse_mean,
                        'r_mean':float(np.mean([s['r'] for s in scores])),'p_mean':float(np.mean([s['p'] for s in scores])),
                        'r_overall':r_all,'p_overall':p_all}])
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Experiment 3: metadata combinations")
    ap.add_argument('--inject', choices=['clip','turn'], default='clip')
    ap.add_argument('--backbone', choices=['hybrid','lstm'], default='hybrid')
    ap.add_argument('--log_duration', action='store_true')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    # 预设组合；也支持自定义
    ap.add_argument('--preset', choices=['all','paper','minimal'], default='paper',
                    help="paper: Gender/Duration/Item/Duration+Gender/All; minimal: Gender, Duration(log), Item; all: 扩展含二元列")
    ap.add_argument('--custom', type=str, default='', help="自定义组合，用;分隔多组，每组用,分隔列名，如: 'Gender;Duration;Item,Duration'")
    args=ap.parse_args()

    # 组合集
    combos=[]
    if args.custom.strip():
        for grp in args.custom.split(';'):
            cols=[t.strip() for t in grp.split(',') if t.strip()]
            if cols: combos.append(cols)
    else:
        if args.preset=='minimal':
            combos=[['Gender'], ['Duration'], ['Item']]
        elif args.preset=='paper':
            combos=[['Gender'], ['Duration'], ['Item'],
                    ['Gender','Duration'], ['Gender','Item','Duration']]
        elif args.preset=='all':
            combos=[['Gender'], ['Duration'], ['Item'],
                    ['R'], ['C'], ['Consensus'],
                    ['Gender','Duration'], ['Item','Duration'],
                    ['Gender','Item','Duration'], ['Gender','Item','Duration','R','C','Consensus']]

    ROOT = os.path.join(os.path.dirname(BASE), 'Results', 'exp3')
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    for combo in combos:
        tag = "_".join([c.replace(' ','') for c in combo]).lower()
        out = os.path.join(ROOT, tag)
        print(f"[RUN exp3] backbone={args.backbone} inject={args.inject} combo={combo} -> {out}")
        run_cv(args.inject, combo, out, log_duration=args.log_duration, epochs=args.epochs, batch=args.batch, lr=args.lr, backbone=args.backbone)
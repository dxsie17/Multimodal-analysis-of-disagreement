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

def write_summary(out_dir, mode_tag, meta_tag, scores, overall_r=None, overall_p=None):
    mse_mean = float(np.mean([s['MSE'] for s in scores]))
    mae_mean = float(np.mean([s['MAE'] for s in scores]))
    r2_mean  = float(np.mean([s['R2']  for s in scores]))
    rmse_mean = float(np.mean([np.sqrt(s['MSE']) for s in scores]))
    df = pd.DataFrame([{
        'Mode': mode_tag, 'Meta': meta_tag or '',
        'MSE_mean': mse_mean, 'MAE_mean': mae_mean, 'R2_mean': r2_mean, 'RMSE_mean': rmse_mean,
        'r_mean': float(np.mean([s.get('r', np.nan) for s in scores])),
        'p_mean': float(np.nanmean([s.get('p', np.nan) for s in scores])),
        'r_overall': overall_r if overall_r is not None else np.nan,
        'p_overall': overall_p if overall_p is not None else np.nan,
    }])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

# ---- Load data ----
data  = np.load(NPZ, allow_pickle=True)
X, y, clips = data["X"].astype("float32"), data["y"].astype("float32"), data["clips"]
meta = pd.read_csv(META, encoding="latin-1")
key  = "File name" if "File name" in meta.columns else "File_name"
folds = meta.set_index(key).loc[clips, "Fold"].values
meta_df = meta.set_index(key).loc[clips]

L = (np.abs(X).sum(axis=2) > 0).sum(axis=1).astype("int64")

# ---- Meta building ----
def parse_meta_list(meta_list, cols):
    TOK2COL = {"item":"Item","gender":"Gender","duration":"Duration",
               "r":"R Choice","rchoice":"R Choice","c":"C Choice","cchoice":"C Choice","consensus":"Consensus"}
    if meta_list.lower() in ("none","no","off"): selected=[]
    elif meta_list.lower() in ("base","basic"): selected=["Item","Gender","Duration"]
    elif meta_list.lower() in ("all","full"):   selected=["Item","Gender","Duration","R Choice","C Choice","Consensus"]
    else: selected=[TOK2COL.get(t.strip().lower(), t.strip()) for t in meta_list.split(",")]
    selected=[c for c in selected if c in cols]
    cat_cols=[c for c in ["Item","Gender"] if c in selected]
    bin_cols=[c for c in ["R Choice","C Choice","Consensus"] if c in selected]
    num_cols=[c for c in ["Duration"] if c in selected]
    return selected, cat_cols, bin_cols, num_cols

def build_meta(tr_idx, te_idx, cat_cols, bin_cols, num_cols, log_duration=False, duration_bins=0):
    df_tr = meta_df.iloc[tr_idx].copy(); df_te = meta_df.iloc[te_idx].copy()
    parts_tr, parts_te = [], []
    # categorical
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        parts_tr.append(ohe.fit_transform(df_tr[cat_cols])); parts_te.append(ohe.transform(df_te[cat_cols]))
    # binary Yes/No
    if bin_cols:
        parts_tr.append((df_tr[bin_cols]=="Yes").astype(int).values)
        parts_te.append((df_te[bin_cols]=="Yes").astype(int).values)
    # numeric (Duration)
    if num_cols:
        s_tr = df_tr[num_cols].astype(float).copy(); s_te = df_te[num_cols].astype(float).copy()
        if log_duration and "Duration" in num_cols:
            s_tr["Duration"]=np.log1p(s_tr["Duration"]); s_te["Duration"]=np.log1p(s_te["Duration"])
        scaler = StandardScaler(); parts_tr.append(scaler.fit_transform(s_tr.values)); parts_te.append(scaler.transform(s_te.values))
    Xtr_meta = np.concatenate(parts_tr, axis=1).astype("float32") if parts_tr else np.zeros((len(tr_idx),0),dtype="float32")
    Xte_meta = np.concatenate(parts_te, axis=1).astype("float32") if parts_te else np.zeros((len(te_idx),0),dtype="float32")
    return Xtr_meta, Xte_meta

# ---- Models ----
class LSTMReg(nn.Module):
    def __init__(self, d_in, hidden=256, layers=2, bidir=True):
        super().__init__()
        self.norm=nn.LayerNorm(d_in)
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidir, dropout=0.2 if layers>1 else 0.0)
        out_dim = hidden*(2 if bidir else 1)
        self.fc  = nn.Linear(out_dim, 1)
    def forward(self, x,lens):
        x=self.norm(x)
        pack=nn.utils.rnn.pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_) = self.rnn(pack)
        h_last = torch.cat([h[-2],h[-1]],dim=1) if self.rnn.bidirectional else h[-1]
        return self.fc(h_last).squeeze(-1)

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
    """Hybrid backbone: txt-branch BiLSTM + aud-branch BiLSTM -> concat seq -> fusion BiLSTM -> head"""
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

class HybridWithMetaClip(nn.Module):
    """Hybrid + clip-level meta: take fusion last state then concat meta -> MLP head"""
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
    """Hybrid + turn-level meta: broadcast meta to each fused timestep before fusion BiLSTM"""
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

def run_fold(mode, tr_idx, te_idx, meta_spec="base", log_duration=False, backbone="lstm"):
    Xtr, Xte = torch.tensor(X[tr_idx]), torch.tensor(X[te_idx])
    ytr, yte = torch.tensor(y[tr_idx]), torch.tensor(y[te_idx])
    Ltr, Lte = torch.tensor(L[tr_idx]), torch.tensor(L[te_idx])

    model=None
    if mode=="none":
        tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr), batch_size=args.batch, shuffle=True)
        te_dl = DataLoader(TensorDataset(Xte, yte, Lte), batch_size=64, shuffle=False)
        if backbone=="lstm":
            model = LSTMReg(d_in=X.shape[2]).to(DEVICE)
        elif backbone=="hybrid":
            d_txt = TEXT_DIM; d_aud = X.shape[2]-d_txt
            model = HybridReg(d_txt=d_txt, d_aud=d_aud, hidden=256, layers=1, bidir=True, dropout=0.1).to(DEVICE)
        else:
            raise ValueError(backbone)
    else:
        selected, cat_cols, bin_cols, num_cols = parse_meta_list(meta_spec, meta_df.columns)
        Mtr_np, Mte_np = build_meta(tr_idx, te_idx, cat_cols, bin_cols, num_cols, log_duration=log_duration)
        Mtr, Mte = torch.tensor(Mtr_np), torch.tensor(Mte_np)
        if backbone=="lstm":
            if mode=="clip":
                model = LSTMRegWithMetaClip(d_in=X.shape[2], m_in=Mtr.shape[1]).to(DEVICE)
            elif mode=="turn":
                model = LSTMRegWithMetaTurn(d_in=X.shape[2], m_in=Mtr.shape[1]).to(DEVICE)
            else:
                raise ValueError(mode)
        elif backbone=="hybrid":
            d_txt = TEXT_DIM; d_aud = X.shape[2]-d_txt
            if mode=="clip":
                model = HybridWithMetaClip(d_txt=d_txt, d_aud=d_aud, m_in=Mtr.shape[1], hidden=256, layers=1, bidir=True, dropout=0.1).to(DEVICE)
            elif mode=="turn":
                model = HybridWithMetaTurn(d_txt=d_txt, d_aud=d_aud, m_in=Mtr.shape[1], hidden=256, layers=1, bidir=True, dropout=0.1).to(DEVICE)
            else:
                raise ValueError(mode)
        else:
            raise ValueError(backbone)
        tr_dl = DataLoader(TensorDataset(Xtr, ytr, Ltr, Mtr), batch_size=args.batch, shuffle=True)
        te_dl = DataLoader(TensorDataset(Xte, yte, Lte, Mte), batch_size=64, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, wait, patience = float('inf'), 0, 5
    preds=trues=None

    for ep in range(args.epochs):
        model.train()
        for batch in tr_dl:
            if mode=="none":
                xb,yb,lb = [t.to(DEVICE) for t in batch]
                pb = model(xb, lb)
            else:
                xb,yb,lb,mb = [t.to(DEVICE) for t in batch]
                pb = model(xb, lb, mb)
            loss = lossf(pb, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); P=[]; T=[]
        with torch.no_grad():
            for batch in te_dl:
                if mode=="none":
                    xb,yb,lb = [t.to(DEVICE) for t in batch]
                    p = model(xb, lb).cpu().numpy()
                else:
                    xb,yb,lb,mb = [t.to(DEVICE) for t in batch]
                    p = model(xb, lb, mb).cpu().numpy()
                P.append(p); T.append(batch[1].cpu().numpy())
        preds, trues = np.concatenate(P), np.concatenate(T)
        mse = mean_squared_error(trues, preds)
        if mse + 1e-6 < best: best, wait = mse, 0
        else:
            wait += 1
            if wait >= patience: break

    r,p = pearsonr_with_p(trues, preds)
    return {'MSE':mean_squared_error(trues,preds),'MAE':mean_absolute_error(trues,preds),
            'R2':r2_score(trues,preds),'r':r,'p':p,'preds':preds,'trues':trues}

def run_cv(mode, out_dir, meta_spec="base", log_duration=False, backbone="lstm"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    scores, all_rows = [], []
    for k in sorted(np.unique(folds)):
        te_idx = np.where(folds==k)[0]; tr_idx = np.where(folds!=k)[0]
        res = run_fold(mode, tr_idx, te_idx, meta_spec=meta_spec, log_duration=log_duration, backbone=backbone)
        scores.append({m:res[m] for m in ['MSE','MAE','R2','r','p']})
        rmse_k = float(np.sqrt(res['MSE']))
        tag = {'none':'NONE','clip':'CLIP','turn':'TURN'}[mode]
        print(f"[{tag}] Fold {k}: MSE={res['MSE']:.3f}, RMSE={rmse_k:.3f}, MAE={res['MAE']:.3f}, R2={res['R2']:.3f}, r={res['r']:.3f}, p={res['p']:.3g}")
        dfk = pd.DataFrame({'clip':clips[te_idx],'y_true':res['trues'],'y_pred':res['preds'],'fold':k})
        dfk.to_csv(os.path.join(out_dir, f"{mode}_preds_fold{k}.csv"), index=False)
        all_rows.append(dfk)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, f"{mode}_preds_all_folds.csv"), index=False)
    r_all,p_all = pearsonr_with_p(all_df['y_true'].values, all_df['y_pred'].values)
    print(f"[{mode}] Overall across folds: r={r_all:.3f}, p={p_all:.3g}")
    if mode=="none":
        tag = backbone
        meta_tag = ''
    else:
        tag = f"{backbone}_meta_{mode}"
        meta_tag = meta_spec
    write_summary(out_dir, mode_tag=tag, meta_tag=meta_tag, scores=scores, overall_r=r_all, overall_p=p_all)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experiment 2: metadata injection (none/clip/turn)")
    ap.add_argument('--inject', choices=['all','none','clip','turn'], default='all')
    ap.add_argument('--meta', type=str, default='base', help="none/base/all or comma list: Item,Gender,Duration,...")
    ap.add_argument('--log_duration', action='store_true', help='log1p for Duration if used')
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone', choices=['lstm','hybrid'], default='lstm', help='fix model backbone for exp2')
    args = ap.parse_args()

    ROOT = os.path.join(os.path.dirname(BASE), "Results", "exp2")
    if args.inject in ('all','none'):
        out = os.path.join(ROOT, 'none');   print(f"[RUN exp2] injection=none  -> {out}")
        run_cv('none', out, backbone=args.backbone)
    if args.inject in ('all','clip'):
        out = os.path.join(ROOT, 'clip');   print(f"[RUN exp2] injection=clip  -> {out}")
        run_cv('clip', out, meta_spec=args.meta, log_duration=args.log_duration, backbone=args.backbone)
    if args.inject in ('all','turn'):
        out = os.path.join(ROOT, 'turn');   print(f"[RUN exp2] injection=turn  -> {out}")
        run_cv('turn', out, meta_spec=args.meta, log_duration=args.log_duration, backbone=args.backbone)
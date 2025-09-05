import os
import math
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error

from build_dataset import DisagreeDataset, T_MAX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Model --------------------
class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0), bidirectional=bidirectional)
        self.hidden = hidden
        self.bidirectional = bidirectional

    @property
    def out_dim(self) -> int:
        return self.hidden * (2 if self.bidirectional else 1)

    def forward(self, x, lengths):
        # x: (B, T, D)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T_MAX)
        # Use last valid timestep per sample as clip representation
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(1, idx).squeeze(1)  # (B, H)
        return out, last


class BiLSTMRegressor(nn.Module):
    def __init__(self, d_txt: int, d_aud: int, d_meta: int, fusion: str = "early", meta_mode: str = "none",
                 hidden: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        assert fusion in {"early", "late", "hybrid"}
        assert meta_mode in {"none", "clip", "turn"}
        self.fusion = fusion
        self.meta_mode = meta_mode

        # Input dims per mode
        d_turn_meta = d_meta if meta_mode == "turn" else 0

        if fusion == "early":
            self.branch = BiLSTMBlock(d_txt + d_aud + d_turn_meta, hidden, num_layers, dropout)
            head_in = self.branch.out_dim + (d_meta if meta_mode == "clip" else 0)
        elif fusion == "late":
            self.txt_branch = BiLSTMBlock(d_txt + d_turn_meta, hidden, num_layers, dropout)
            self.aud_branch = BiLSTMBlock(d_aud + d_turn_meta, hidden, num_layers, dropout)
            head_in = self.txt_branch.out_dim + self.aud_branch.out_dim + (d_meta if meta_mode == "clip" else 0)
        else:  # hybrid: separate -> concat -> fusion BiLSTM
            self.txt_branch = BiLSTMBlock(d_txt + d_turn_meta, hidden // 2, num_layers, dropout)
            self.aud_branch = BiLSTMBlock(d_aud + d_turn_meta, hidden // 2, num_layers, dropout)
            fused_in = self.txt_branch.out_dim + self.aud_branch.out_dim
            self.fusion_lstm = BiLSTMBlock(fused_in, hidden, num_layers, dropout)
            head_in = self.fusion_lstm.out_dim + (d_meta if meta_mode == "clip" else 0)

        self.regressor = nn.Sequential(
            nn.Linear(head_in, head_in // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_in // 2, 1)
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        txt = batch["txt"]  # (B,T,Dt)
        aud = batch["aud"]  # (B,T,Da)
        mask = batch["mask"]  # (B,T)
        lengths = mask.sum(dim=1).long()
        meta_turn = batch["meta_turn"]  # (B,T,Dm_t) or empty
        meta_clip = batch["meta_clip"]  # (B,Dm_c) or empty

        def add_turn_meta(x):
            if meta_turn.numel() == 0:
                return x
            return torch.cat([x, meta_turn], dim=-1)

        if self.fusion == "early":
            x = torch.cat([txt, aud], dim=-1)
            x = add_turn_meta(x)
            _, h = self.branch(x, lengths)
        elif self.fusion == "late":
            x_txt = add_turn_meta(txt)
            x_aud = add_turn_meta(aud)
            _, h_txt = self.txt_branch(x_txt, lengths)
            _, h_aud = self.aud_branch(x_aud, lengths)
            h = torch.cat([h_txt, h_aud], dim=-1)
        else:  # hybrid
            x_txt = add_turn_meta(txt)
            x_aud = add_turn_meta(aud)
            seq_txt, h_txt = self.txt_branch(x_txt, lengths)
            seq_aud, h_aud = self.aud_branch(x_aud, lengths)
            fused_seq = torch.cat([seq_txt, seq_aud], dim=-1)
            _, h = self.fusion_lstm(fused_seq, lengths)

        if meta_clip.numel() > 0:
            h = torch.cat([h, meta_clip], dim=-1)
        y = self.regressor(h).squeeze(-1)
        return y


# -------------------- Metrics --------------------
def rmse(y_true, y_pred):
    return math.sqrt(float(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# -------------------- Training --------------------
def run_fold(fold_id: int, args):
    # train on folds != fold_id, validate on fold_id
    train_folds = [f for f in range(5) if f != fold_id]
    valid_folds = [fold_id]

    meta_cfg = {
        "use_gender": args.meta_combo in {"gender", "gender_duration", "all"},
        "use_item": args.meta_combo in {"item", "all"},
        "use_duration": args.meta_combo in {"duration", "gender_duration", "all"},
        "duration_log": True,
    }

    ds_tr = DisagreeDataset(train_folds, fusion=args.fusion, meta_mode=args.meta_mode, meta_cfg=meta_cfg)
    ds_va = DisagreeDataset(valid_folds, fusion=args.fusion, meta_mode=args.meta_mode, meta_cfg=meta_cfg)

    # infer dims from first sample
    s = ds_tr[0]
    d_txt = s["txt"].shape[-1]
    d_aud = s["aud"].shape[-1]
    d_meta_turn = s["meta_turn"].shape[-1]
    d_meta_clip = s["meta_clip"].shape[-1]
    d_meta = max(d_meta_turn, d_meta_clip)

    model = BiLSTMRegressor(d_txt, d_aud, d_meta, fusion=args.fusion, meta_mode=args.meta_mode,
                            hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(DEVICE)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"rmse": 1e9, "state": None}

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            for k in ["txt", "aud", "mask", "meta_turn", "meta_clip", "y"]:
                batch[k] = batch[k].to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = crit(pred, batch["y"])  # regression MSE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in valid_loader:
                for k in ["txt", "aud", "mask", "meta_turn", "meta_clip", "y"]:
                    batch[k] = batch[k].to(DEVICE)
                pred = model(batch)
                ys.append(batch["y"].cpu().numpy())
                ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        metrics = evaluate(y_true, y_pred)
        if metrics["RMSE"] < best["rmse"]:
            best = {"rmse": metrics["RMSE"], "state": model.state_dict()}
        print(f"Fold {fold_id} | Epoch {epoch+1}/{args.epochs} | "
              f"RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}")

    return best["rmse"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", choices=["early", "late", "hybrid"], default="early")
    p.add_argument("--meta_mode", choices=["none", "clip", "turn"], default="none")
    p.add_argument("--meta_combo", choices=["none", "gender", "duration", "item", "gender_duration", "all"], default="none")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    # --------- Three experiment groups (per thesis design) ---------
    # A. 融合方式对比（无元数据） → 运行三次：early/late/hybrid + meta_mode=none
    if args.fusion == "grid_exp1":
        combos = [
            dict(fusion="early", meta_mode="none", meta_combo="none"),
            dict(fusion="late", meta_mode="none", meta_combo="none"),
            dict(fusion="hybrid", meta_mode="none", meta_combo="none"),
        ]
    # B. 元数据注入方式对比（固定融合方式，用 --fusion 固定）
    elif args.fusion == "grid_exp2":
        base_fusion = "early"
        combos = [
            dict(fusion=base_fusion, meta_mode="none", meta_combo="none"),
            dict(fusion=base_fusion, meta_mode="clip", meta_combo="gender_duration"),
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="gender_duration"),
        ]
    # C. 元数据组合对比（固定注入方式 turn-level）
    elif args.fusion == "grid_exp3":
        base_fusion = "early"
        combos = [
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="gender"),
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="duration"),
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="item"),
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="gender_duration"),
            dict(fusion=base_fusion, meta_mode="turn", meta_combo="all"),
        ]
    else:
        combos = [dict(fusion=args.fusion, meta_mode=args.meta_mode, meta_combo=args.meta_combo)]

    for cfg in combos:
        print("==== Run:", cfg)
        # Map cfg into args-like container
        class C: pass
        ca = C()
        ca.fusion = cfg["fusion"]
        ca.meta_mode = cfg["meta_mode"]
        ca.meta_combo = cfg["meta_combo"]
        ca.hidden = args.hidden
        ca.layers = args.layers
        ca.dropout = args.dropout
        ca.batch_size = args.batch_size
        ca.epochs = args.epochs
        ca.lr = args.lr

        rmses = []
        for f in range(5):
            rmse_best = run_fold(f, ca)
            rmses.append(rmse_best)
        print("CV RMSE per fold:", rmses)
        print("Mean RMSE:", float(np.mean(rmses)))


if __name__ == "__main__":
    main()
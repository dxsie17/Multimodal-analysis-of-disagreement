import os, numpy as np, pandas as pd
from glob import glob

BASE = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset"
TXT_DIR  = os.path.join(BASE, "Embedding", "txt_embeddings")
AUD_DIR  = os.path.join(BASE, "Embedding", "audio_embeddings")
META_CSV = os.path.join(BASE, "metadata.csv")

MAX_LEN = 20
OUT_NPZ = os.path.join(BASE, "Embedding", f"dataset_T{MAX_LEN}.npz")

def load_txt_matrix(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, encoding="latin-1")
    if any(c.startswith("txt_emb_") for c in df.columns):
        return df.filter(like="txt_emb_").values
    meta_cols = [c for c in ["start", "end", "speaker", "text"] if c in df.columns]
    return df.drop(columns=meta_cols).values

def pad_or_trunc(seq: np.ndarray, max_len: int) -> np.ndarray:
    if seq.shape[0] >= max_len:
        return seq[:max_len]
    pad = np.zeros((max_len - seq.shape[0], seq.shape[1]), dtype=seq.dtype)
    return np.vstack([seq, pad])

def main():
    meta = pd.read_csv(META_CSV, encoding="latin-1")
    txt_files = sorted(glob(os.path.join(TXT_DIR, "*_embedding.csv")))
    X_list, y_list, clips = [], [], []

    for tf in txt_files:
        clip = os.path.basename(tf).replace("_embedding.csv", "")
        af   = os.path.join(AUD_DIR, f"{clip}.npy")
        if not os.path.exists(af):
            print(f"[!] skip: audio npy not found for {clip}")
            continue

        txt_feats = load_txt_matrix(tf)          # (T, d_txt)
        aud_feats = np.load(af)                  # (T, d_aud)
        if txt_feats.shape[0] != aud_feats.shape[0]:
            print(f"[!] skip: turn mismatch for {clip}: text {txt_feats.shape[0]} vs audio {aud_feats.shape[0]}")
            continue

        seq = np.concatenate([txt_feats, aud_feats], axis=1)   # (T, d_txt+d_aud)
        seq = pad_or_trunc(seq, MAX_LEN)                       # (MAX_LEN, D)
        X_list.append(seq)

        # Rating
        row = meta.loc[meta["File name"] == clip] if "File name" in meta.columns else meta.loc[meta["File_name"] == clip]
        if row.empty:
            print(f"[!] skip: label not found for {clip}")
            continue
        y_list.append(float(row["Rating"].values[0]))
        clips.append(clip)

    X = np.stack(X_list) if X_list else np.zeros((0, MAX_LEN, 1))
    y = np.array(y_list, dtype=float)
    np.savez_compressed(OUT_NPZ, X=X, y=y, clips=np.array(clips))
    print("Saved:", OUT_NPZ, "| X:", X.shape, "y:", y.shape, "clips:", len(clips))

if __name__ == "__main__":
    main()
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd

def extract_audio_embeddings(
    audio_dir: str = "disagreement-dataset/Audio",
    out_dir:   str = "disagreement-dataset/Embedding/audio_embeddings",
    model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
):
    # Load the model with the processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    # Compute minimum required segment length to satisfy convolutional kernel sizes
    conv_kernel_sizes = [layer.conv.kernel_size[0] for layer in model.feature_extractor.conv_layers]
    min_seg_len = max(conv_kernel_sizes)

    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(audio_dir):
        if not fname.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(audio_dir, fname)
        # Read and resample to 16kHz
        speech, sr = librosa.load(wav_path, sr=16000)

        # Load corresponding Turn-level transcripts
        base, _ = os.path.splitext(fname)
        csv_path = os.path.join("disagreement-dataset", "Transcriptions", "Turn_Level_Transcripts", f"{base}.csv")
        df = pd.read_csv(csv_path, encoding="latin-1")
        audio_embs = []
        # Preload full audio
        full_audio, _ = librosa.load(wav_path, sr=16000)
        for _, row in df.iterrows():
            start, end = row['start'], row['end']
            s_i, e_i = int(start * sr), int(end * sr)
            segment = full_audio[s_i:e_i]
            # Ensure segment is at least min_seg_len samples long
            if segment.shape[0] < min_seg_len:
                pad_width = min_seg_len - segment.shape[0]
                segment = np.pad(segment, (0, pad_width), mode='constant', constant_values=0)
            inputs = processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state  # (1, T_seg, D)
                emb = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            except RuntimeError as e:
                # If segment too short for convolution, fallback to zero vector
                emb = np.zeros(model.config.hidden_size, dtype=float)
                print(f"[!] Skipped short segment {base} ({start:.2f}-{end:.2f}s): {e}")
            audio_embs.append(emb)
        # Save per-turn embeddings as a numpy array
        audio_embs_array = np.vstack(audio_embs)  # shape: (num_turns, D)
        out_path = os.path.join(out_dir, f"{base}.npy")
        np.save(out_path, audio_embs_array)
        print(f"[√] Saved per-turn embeddings for {fname} → {out_path}")

if __name__ == "__main__":
    extract_audio_embeddings()
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

import os

INPUT_DIR = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset/Transcriptions/Turn_Level_Transcripts"
OUTPUT_DIR = "/Users/susie/Desktop/Multimodal analysis of disagreement/disagreement-dataset/Embedding/txt_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / summed_mask

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".csv"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", "_embedding.csv"))
        
        df = pd.read_csv(input_path, encoding='latin-1')
        embeddings = []

        for text in tqdm(df['text'], desc=f"Embedding {filename}"):
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            sentence_embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(sentence_embedding.squeeze().numpy())

        emb_df = pd.DataFrame(embeddings)
        result = pd.concat([df[['start', 'end', 'speaker', 'text']], emb_df], axis=1)
        result.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

"""
build_index.py
- Load your PubMed dataset (CSV/TSV/JSONL) with columns at least: id, title, abstract
- Compute embeddings (sentence-transformers)
- Build a FAISS index and save index + metadata

Usage:
python build_index.py --input pubmed.csv --id_col pmid --title_col title --abstract_col abstract --out_dir ./pubmed_index
"""
import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm

def load_dataset(path: str, id_col='id', title_col='title', abstract_col='abstract'):
    _, ext = os.path.splitext(path)
    if ext in ['.csv', '.tsv']:
        sep = ',' if ext == '.csv' else '\t'
        df = pd.read_csv(path, sep=sep)
    elif ext in ['.json', '.jsonl']:
        df = pd.read_json(path, lines=(ext=='.jsonl'))
    else:
        raise ValueError("Unsupported file extension. Use csv/tsv/json/jsonl")
    assert id_col in df.columns and abstract_col in df.columns
    # Keep only records where abstract exists
    df = df[~df[abstract_col].isnull()].reset_index(drop=True)
    return df

def chunk_text(text, max_len=1000):
    # naive chunker: split by sentences up to a char threshold
    if len(text) <= max_len:
        return [text]
    chunks = []
    sentences = text.split('. ')
    cur = ''
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + '. ' + s).strip() if cur else s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def build(args):
    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading dataset...")
    df = load_dataset(args.input, id_col=args.id_col, title_col=args.title_col, abstract_col=args.abstract_col)

    print(f"Using embedding model: {args.emb_model}")
    embedder = SentenceTransformer(args.emb_model)

    texts = []
    meta = []  # list of dicts for each chunk: {id, title, chunk_id, text}
    idx = 0
    print("Chunking and collecting texts...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(row[args.id_col])
        title = str(row[args.title_col]) if args.title_col in row else ""
        abstract = str(row[args.abstract_col])
        chunks = chunk_text(abstract, max_len=args.chunk_max_chars)
        for i, c in enumerate(chunks):
            texts.append(c)
            meta.append({'doc_id': doc_id, 'title': title, 'chunk_id': i, 'text': c})
            idx += 1

    print("Computing embeddings in batches...")
    batch_size = args.batch_size
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')

    d = embeddings.shape[1]
    print(f"Embedding dimension: {d}, num vectors: {embeddings.shape[0]}")

    # Build FAISS index
    print("Building FAISS index (IndexFlatIP + normalize for cosine)...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))

    # Save metadata
    print("Saving metadata...")
    with open(os.path.join(args.out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("Done. Index saved to", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to pubmed file (csv/json/jsonl)")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--title_col", default="title")
    parser.add_argument("--abstract_col", default="abstract")
    parser.add_argument("--out_dir", default="./pubmed_index")
    parser.add_argument("--emb_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_max_chars", type=int, default=1000)
    args = parser.parse_args()
    build(args)

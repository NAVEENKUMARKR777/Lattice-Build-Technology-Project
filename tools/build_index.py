import argparse
import glob
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

EMBED_MODEL = "intfloat/e5-small-v2"


def load_documents(json_dir):
    """Return list of (doc_id, text)"""
    docs = []
    for fp in glob.glob(str(Path(json_dir) / "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            text = f"{it['title']} \n {it['summary']}"
            docs.append((it["id"], text))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG")
    parser.add_argument("--data_dir", type=str, default="data/papers", help="Where JSON files are stored")
    parser.add_argument("--index_dir", type=str, default="data/index", help="Output directory for FAISS index + metadata")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    docs = load_documents(args.data_dir)
    print(f"Loaded {len(docs)} documents")

    model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    embeddings = []
    ids = []
    for i in tqdm(range(0, len(docs), args.batch), desc="Embedding"):
        batch_text = [t[1] for t in docs[i : i + args.batch]]
        batch_emb = model.encode(batch_text, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(batch_emb)
        ids.extend([t[0] for t in docs[i : i + args.batch]])

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and metadata
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "quant_ph.index"))
    with open(out_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print("[+] Index saved to", out_dir)


if __name__ == "__main__":
    main() 
import argparse
import json
# Optional: GNU readline is Unix-only. Silently ignore on Windows.
try:
    import readline  # type: ignore  # noqa: F401
except ImportError:
    pass
from pathlib import Path
import sys

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

EMBED_MODEL = "intfloat/e5-small-v2"
DEFAULT_BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"


class RAGEngine:
    def __init__(self, index_dir: str, lora_dir: str | None = None, top_k: int = 5):
        # Embedding model
        self.embed_model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

        # Load FAISS index + metadata
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "quant_ph.index"))
        with open(index_dir / "ids.json", "r", encoding="utf-8") as f:
            self.ids = json.load(f)

        # For mapping id-> text we need documents themselves
        self.doc_map = {}
        for json_file in Path("data/papers").glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                for doc in json.load(f):
                    self.doc_map[doc["id"]] = f"{doc['title']}\n{doc['summary']}"

        # Language model – try 4-bit first on non-Windows, otherwise fallback to FP16
        base_model = None
        if sys.platform != "win32":
            try:
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                base_model = AutoModelForCausalLM.from_pretrained(
                    DEFAULT_BASE_MODEL,
                    device_map="auto",
                    quantization_config=quant_cfg,
                )
                print("[✓] Loaded 4-bit model with bitsandbytes")
            except Exception as e:
                print("[!] 4-bit load failed, fallback to 16-bit. Reason:", e)

        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_BASE_MODEL,
                torch_dtype=torch.float16,
            ).to("cuda")

        base_model.config.use_cache = False  # for consistency with gradient-ckpt models

        if lora_dir is not None:
            print(f"[+] Loading LoRA adapters from {lora_dir}")
            base_model = PeftModel.from_pretrained(base_model, lora_dir)

        self.model = base_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.top_k = top_k

    def retrieve(self, query: str):
        q_emb = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(np.expand_dims(q_emb, axis=0), self.top_k)
        results = [self.doc_map[self.ids[i]] for i in idxs[0]]
        return results

    def chat(self):
        print("Type your question (Ctrl+C to exit).\n")
        while True:
            try:
                user_q = input(">>> ").strip()
                if not user_q:
                    continue
                contexts = self.retrieve(user_q)
                prompt = self.build_prompt(user_q, contexts)
                response = self.generate(prompt)
                print("\n" + response + "\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting…")
                break

    def build_prompt(self, question: str, contexts: list[str]):
        joined_context = "\n---\n".join(contexts)
        prompt = (
            "<|system|>You are an expert quantum physics assistant. Answer using the given context.\n"
            f"<|context|>\n{joined_context}\n"
            f"<|user|> {question}\n<|assistant|>"
        )
        return prompt

    def generate(self, prompt: str, max_tokens: int = 256):
        enc = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        output_text = self.tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="data/index")
    parser.add_argument("--lora_dir", type=str, default="models/phi3_quant")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    rag = RAGEngine(args.index_dir, args.lora_dir, args.top_k)
    rag.chat() 
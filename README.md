# Quantum Physics RAG with Phi-3

This repo contains a minimal, fully–local prototype that combines:

1. Retrieval-Augmented Generation (RAG) over the latest quantum-physics papers from arXiv.
2. Parameter-efficient fine-tuning (LoRA) of Microsoft's **Phi-3 Mini** language model on those papers.
3. A CLI chat interface that answers quantum-physics questions using the tuned model + retrieval context.

The entire stack runs **locally** – no hosted APIs are required.

---

## 1. Quick start

```bash
# 1. Create virtual environment (Windows-friendly example)
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch the latest papers (defaults to 50 most recent in `quant-ph`)
python tools/download_papers.py -n 500 -o data/papers

# 4. Fine-tune Phi-3 with LoRA (adjust hyper-parameters in `scripts/fine_tune.py`)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python tools/fine_tune.py --data_dir data/papers --output_dir models/phi3_quant --batch 1

# 5. Build the FAISS vector store for RAG
python tools/build_index.py --data_dir data/papers --index_dir data/index

# 6. Chat! 🎉
python tools/rag_chat.py --index_dir data/index --lora_dir models/phi3_quant
```

> **Tip**: The workflow is modular – you can re-run any step independently.

---

## 2. Directory layout

```
├── data/
│   ├── papers/          # Raw JSON dumps from arXiv
│   └── index/           # FAISS vectors + metadata
├── models/
│   └── phi3_quant/      # LoRA adapters + tokenizer config
├── tools/
│   ├── download_papers.py
│   ├── build_index.py
│   ├── fine_tune.py
│   └── rag_chat.py
├── requirements.txt
└── README.md
```

---

## 3. Hardware requirements

* **GPU strongly recommended** – Phi-3 Mini (2.7 B params) fits comfortably on a single consumer GPU when loaded in 4-bit quantised form. If no GPU is detected the scripts will fall back to CPU, but training/inference will be slow.
* Tested with CUDA 12 + PyTorch 2.1.

---

## 4. Troubleshooting

* **'No module named faiss'** – Ensure you installed the `faiss-cpu` (or `faiss-gpu`) wheel that matches your Python version.
* **Out-of-memory during fine-tuning** – Lower `--per_device_train_batch_size` or switch to CPU training.
* **Windows compile errors** – Use the pre-built wheels in `requirements.txt`; avoid manually building FAISS/bitsandbytes on Windows.

---

## 5. Acknowledgements

* [Microsoft Phi-3](https://aka.ms/phi-3) – lightweight LLM with improved reasoning.
* [arXiv API](https://arxiv.org/help/api/) for open-access scientific papers.
* HuggingFace ecosystem: `transformers`, `datasets`, `peft`, `sentence-transformers`. 
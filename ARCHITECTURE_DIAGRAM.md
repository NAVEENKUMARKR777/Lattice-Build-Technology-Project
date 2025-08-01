# System Architecture Diagram

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTUM PHYSICS RAG SYSTEM                              │
│                         (Fully Local Deployment)                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA PIPELINE │    │  MODEL TRAINING │    │  RAG INTERFACE  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │arXiv API    │ │    │ │Phi-3 Mini   │ │    │ │User Query   │ │
│ │quant-ph     │ │    │ │2.7B params  │ │    │ │             │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│        │        │    │        │        │    │        │        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │JSON Papers  │ │    │ │LoRA Adapters│ │    │ │Semantic     │ │
│ │2,900 docs   │ │    │ │16M params   │ │    │ │Search       │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    │        │        │
        │                       │             │ ┌─────────────┐ │
        │                       │             │ │Context      │ │
        │                       │             │ │Retrieval    │ │
        │                       │             │ └─────────────┘ │
        │                       │             │        │        │
        │                       │             │ ┌─────────────┐ │
        │                       │             │ │Fine-tuned   │ │
        │                       │             │ │Response     │ │
        │                       │             │ └─────────────┘ │
        │                       │             └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VECTOR DB     │    │   MODEL STORE   │    │   USER OUTPUT   │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │FAISS Index  │ │    │ │LoRA Weights │ │    │ │Expert       │ │
│ │E5 Embeddings│ │    │ │12MB         │ │    │ │Response     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Metadata     │ │    │ │Tokenizer    │ │    │ │Citations    │ │
│ │Mapping      │ │    │ │Config       │ │    │ │Sources      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Detailed Component Architecture

### 1. Data Collection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION STAGE                       │
└─────────────────────────────────────────────────────────────────┘

arXiv API (quant-ph category)
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    download_papers.py                          │
├─────────────────────────────────────────────────────────────────┤
│ • arxiv.Search(query="cat:quant-ph")                          │
│ • max_results=2900                                             │
│ • sort_by=SubmittedDate                                        │
│ • sort_order=Descending                                        │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                             │
├─────────────────────────────────────────────────────────────────┤
│ • Title extraction                                             │
│ • Abstract extraction                                          │
│ • Metadata (ID, URL, date)                                    │
│ • JSON serialization                                           │
│ • Timestamped file naming                                      │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: quant_ph_YYYYMMDD_HHMMSS.json       │
│                    Size: 3.8MB, Papers: 2,900                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Model Fine-tuning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL FINE-TUNING STAGE                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    fine_tune.py                                │
├─────────────────────────────────────────────────────────────────┤
│ • Base Model: microsoft/phi-3-mini-4k-instruct                │
│ • Quantization: 4-bit (bitsandbytes)                          │
│ • LoRA Configuration:                                          │
│   - r=16, alpha=32, dropout=0.05                              │
│   - target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj]│
│ • Training: 1 epoch, 184 steps                                │
│ • Loss: 2.331 → 2.206 (5.4% improvement)                      │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                          │
├─────────────────────────────────────────────────────────────────┤
│ • Text concatenation: title + "\n\n" + summary                │
│ • Tokenization with Phi-3 tokenizer                           │
│ • Chunking: 256 tokens per sequence                           │
│ • EOS token addition                                          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING OPTIMIZATION                       │
├─────────────────────────────────────────────────────────────────┤
│ • Gradient checkpointing                                      │
│ • Mixed precision (FP16/BF16)                                 │
│ • Effective batch size: 16                                    │
│ • Learning rate: 2e-4 with decay                              │
│ • Memory optimization for consumer GPUs                       │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: LoRA Adapters                       │
│                    Size: 12MB, Trainable params: 16M           │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Vector Database Construction

```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE STAGE                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    build_index.py                              │
├─────────────────────────────────────────────────────────────────┤
│ • Embedding Model: intfloat/e5-small-v2                       │
│ • Dimension: 384                                              │
│ • Similarity: Inner Product (cosine)                          │
│ • Index Type: FAISS IndexFlatIP                               │
│ • Batch Processing: 64 documents                              │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                        │
├─────────────────────────────────────────────────────────────────┤
│ • Text format: title + "\n" + summary                         │
│ • SentenceTransformer encoding                                │
│ • Normalized embeddings                                       │
│ • GPU acceleration (when available)                           │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INDEX STORAGE                               │
├─────────────────────────────────────────────────────────────────┤
│ • FAISS binary index: quant_ph.index                          │
│ • Metadata mapping: ids.json                                  │
│ • Document mapping: title + summary                           │
│ • Efficient similarity search                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. RAG Chat Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG CHAT INTERFACE                          │
└─────────────────────────────────────────────────────────────────┘

User Input: "What is quantum entanglement?"
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGEngine.retrieve()                        │
├─────────────────────────────────────────────────────────────────┤
│ • Query embedding with E5 model                               │
│ • FAISS similarity search (top-k=5)                          │
│ • Document retrieval from metadata mapping                    │
│ • Context preparation                                         │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT CONSTRUCTION                          │
├─────────────────────────────────────────────────────────────────┤
│ <|system|>You are an expert quantum physics assistant...       │
│ <|context|>                                                    │
│ [Retrieved paper abstracts]                                    │
│ <|user|> What is quantum entanglement?                         │
│ <|assistant|>                                                  │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAGEngine.generate()                        │
├─────────────────────────────────────────────────────────────────┤
│ • Fine-tuned Phi-3 model                                      │
│ • LoRA adapters loaded                                         │
│ • Context-aware generation                                     │
│ • Temperature: 0.7, Top-p: 0.9                                │
│ • Max tokens: 256                                              │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Expert Response                     │
│                    Grounded in research papers                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   arXiv     │───▶│  Download   │───▶│   Papers    │───▶│   Fine-     │
│   API       │    │  Script     │    │   JSON      │    │   Tune      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │◀───│   Chat      │◀───│   RAG       │◀───│   LoRA      │
│   Query     │    │   Interface │    │   Engine    │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Vector    │    │   FAISS     │
                   │   Index     │    │   Search    │
                   └─────────────┘    └─────────────┘
```

## Memory and Performance Characteristics

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM RESOURCE USAGE                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TRAINING      │    │   INFERENCE     │    │   STORAGE       │
│                 │    │                 │    │                 │
│ GPU Memory:     │    │ GPU Memory:     │    │ Model: 12MB     │
│ ~8GB            │    │ ~6GB            │    │ Index: 8MB      │
│                 │    │                 │    │ Papers: 3.8MB   │
│ Training Time:  │    │ Response Time:  │    │ Total: ~24MB    │
│ ~30 minutes     │    │ <2 seconds      │    │                 │
│                 │    │                 │    │                 │
│ Batch Size: 1   │    │ Top-k: 5        │    │ Scalable to     │
│ Accumulation: 16│    │ Max Tokens: 256 │    │ 10K+ papers     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FRAMEWORKS    │    │   MODELS        │    │   UTILITIES     │
│                 │    │                 │    │                 │
│ • PyTorch 2.1+  │    │ • Phi-3 Mini    │    │ • FAISS         │
│ • Transformers  │    │ • E5-small-v2   │    │ • Sentence      │
│ • PEFT          │    │ • LoRA          │    │   Transformers  │
│ • Accelerate    │    │ • 4-bit Quant   │    │ • ArXiv API     │
│ • Datasets      │    │                 │    │ • BitsAndBytes  │
└─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OPTIMIZATION  │    │   DEPLOYMENT    │    │   COMPATIBILITY │
│                 │    │                 │    │                 │
│ • Gradient      │    │ • Local Only    │    │ • Windows 10+   │
│   Checkpointing │    │ • No APIs       │    │ • Linux         │
│ • Mixed         │    │ • Consumer      │    │ • CUDA 12+      │
│   Precision     │    │   Hardware      │    │ • Python 3.8+   │
│ • LoRA          │    │ • Privacy       │    │ • 8GB+ GPU      │
│ • Quantization  │    │   Preserved     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
``` 


### Step 1 — Download corpora
# 50k PubMed abstracts from HuggingFace Hub (streams Parquet, ~180 MB)
uv run python scripts\download_pubmed_sample.py --max-docs 50000

# Split into disjoint train / held-out (deterministic, seed 42)
uv run python scripts/split_corpus.py

# → data/pubmed_train.jsonl    (45k abstracts)
# → data/pubmed_heldout.jsonl  ( 5k abstracts — never used for training)

### Step 2 — Train the tokenizers
uv run python scripts\train_medical_tokenizer.py --corpus data\pubmed_train.jsonl --out artifacts\medical-bpe-pubmed\tokenizer.json
Equivalent window command: uv run python scripts/train_medical_tokenizer.py --corpus data/pubmed_train.jsonl --vocab-size 16000 --out artifacts/medical-bpe-pubmed

### STEP 3 — Fix/verify the tokenizer directory(Only if directory name is duplicated)

If you haven't completed this successfully yet, the commands are:

Rename-Item artifacts\medical-bpe-pubmed medical-bpe-pubmed.backup
then:
New-Item -ItemType Directory -Path artifacts\medical-bpe-pubmed
then retrain:
uv run python scripts\train_medical_tokenizer.py --corpus data\pubmed_train.jsonl --out artifacts\medical-bpe-pubmed\tokenizer.json

### STEP 4 — Verify that your new tokenizer exists

Test-Path artifacts\medical-bpe-pubmed\tokenizer.json
True

dir artifacts\medical-bpe-pubmed
tokenizer.json

### STEP 5 — Determine vocabulary size

You can inspect the vocabulary size:
print("Vocabulary size:", tokenizer.get_vocab_size())

uv run python scripts\sweep_vocab_size.py --corpus data\pubmed_train.jsonl --heldout data\pubmed_heldout.jsonl --no-qwen

### STEP 6 — Evaluate using the held-out PubMed corpus
pubmed_heldout.jsonl

### STEP 7 — Perform the vocabulary-size sweep
uv run python scripts\sweep_vocab_size.py --corpus data\pubmed_train.jsonl --heldout data\pubmed_heldout.jsonl --no-qwen

### STEP 8 — PubMed tokenizer evaluation
compare_tokenizers.py

### STEP 9 — Wrap your tokenizer into Hugging Face format

Once your final tokenizer has been selected, you can run:

uv run python scripts\wrap_medical_tokenizer.py --tokenizer-json artifacts\medical-bpe-pubmed\tokenizer.json --out artifacts\medical-bpe-pubmed-hf

### STEP 10 — Publish to Hugging Face
Copy-Item .env.example .env
Then open it:
notepad .env
You need to put your Hugging Face write token after the =:
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

### To create a new token in Hugging Face (HF_TOKEN)
--> after login 
--> click profile icon, select settings
--> NAvigate to Tokens, Access Tokens tab
--> Create new token with permission to write, save

uv run python scripts\push_to_hub.py --tokenizer-dir artifacts\medical-bpe-pubmed-hf --repo-id YOUR_USERNAME/medical-bpe-16k








## 📁 Repository layout

```
📦 tokenization-explainer
│
├── 📓 notebooks/
│   ├── 04_custom_vs_general.ipynb   # The lab — 4-way compare + vocab-size experiment
│   └── lab_display.py               # Rich display helpers (chips, tables, charts)
│
├── 📚 docs/
│   ├── theory/                      # 9 slide-ready markdown docs (00–08)
│   │   ├── 00-overview.md           # Lab design, 4-tokenizer table, run order
│   │   ├── 01-why-tokenization-matters.md
│   │   ├── 02-general-purpose-tokenizers.md
│   │   ├── 03-custom-domain-tokenizers.md
│   │   ├── 04-metrics.md            # Fertility, single-token rate, held-out eval
│   │   ├── 05-why-custom-wins-in-healthcare.md
│   │   ├── 06-pretrained-model-trap.md  ⚠️ Read this before any LLM fine-tuning
│   │   ├── 07-optional-lora-sft.md
│   │   └── 08-vocab-size-tradeoff.md
│   └── diagrams/                    # 6 Mermaid diagrams (render on GitHub / JupyterLab)
│       ├── 01-bpe-algorithm.md      # BPE training loop
│       ├── 02-fair-comparison.md    # 2×2 fairness design
│       ├── 03-lab-pipeline.md       # Full pipeline flowchart
│       ├── 04-pretrained-trap.md    # Wrong vs right paths
│       ├── 05-fertility-concept.md  # What fertility means with real numbers
│       └── 06-vocab-size-knee.md    # Diminishing returns + embedding cost
│
├── 🐍 scripts/
│   ├── download_pubmed_sample.py    # Stream PubMed → JSONL (HuggingFace Hub)
│   ├── split_corpus.py              # Deterministic 45k / 5k disjoint split
│   ├── download_general_sample.py   # Stream wikitext-103 → JSONL
│   ├── train_medical_tokenizer.py   # Byte-level BPE trainer
│   ├── wrap_medical_tokenizer.py    # tokenizer.json → PreTrainedTokenizerFast
│   ├── compare_tokenizers.py        # 4-tokenizer CLI with held-out fertility
│   ├── sweep_vocab_size.py          # 16k–100k avg-tokens/doc experiment
│   └── push_to_hub.py               # Upload trained tokenizer to HuggingFace Hub
│
├── 🧪 tests/
│   ├── test_medical_compare.py      # Held-out fairness assertions
│   └── test_vocab_sweep.py          # Knee / band / monotonicity checks
│
├── 🗄️ data/
│   ├── medical_corpus.txt           # Small authored set (tracked)
│   ├── medical_probes.txt           # 20 curated illustration probes (tracked)
│   ├── medical_control.txt          # General English spot-check (tracked)
│   └── *.jsonl                      # Downloaded corpora (gitignored — large)
│
├── 🤗 models/pretrained/
│   └── medical_bpe_tiny/            # Fallback tokenizer if training data absent (tracked)
│
└── 🏺 artifacts/                    # Trained outputs (gitignored)
    ├── medical-bpe-pubmed/          # custom-med 16k
    ├── medical-bpe-pubmed-{16k..100k}/  # vocab-size sweep
    └── general-bpe/                 # fairness control 16k
```

---



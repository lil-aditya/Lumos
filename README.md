# Lumos

Multimodal image retrieval system. Text queries and image uploads get encoded via OpenCLIP, matched against a Qdrant vector index, and served through a FastAPI backend + Streamlit frontend. The whole stack runs in Docker Compose.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-orange.svg)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-00BFFF.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## How It Works

1. **Index** — Images are encoded into 512-d vectors using OpenCLIP (`ViT-B/32`, OpenAI weights) and upserted into Qdrant with cosine similarity.
2. **Search** — A text query or uploaded image is encoded with the same model. Qdrant returns the top-k nearest neighbours.
3. **Serve** — FastAPI exposes search endpoints, Streamlit provides a browser UI.

### Search Modes

| Mode | Input | What happens |
|------|-------|-------------|
| Text → Image | Natural language query | CLIP text encoder → vector search |
| Image → Image | Uploaded image file | CLIP image encoder → vector search |
| Similar by ID | Existing `image_id` in the index | Fetch stored vector → vector search |

---

## Architecture

| Layer | Stack | Detail |
|-------|-------|--------|
| Encoder | OpenCLIP `ViT-B/32` | L2-normalised embeddings, FP16 optional on CUDA |
| Vector DB | Qdrant (REST API) | Cosine distance, batch upsert (512/batch) |
| Backend | FastAPI + Uvicorn | Pydantic request/response models, image proxy |
| Frontend | Streamlit | Tabbed UI — text search, image upload, ID lookup |
| Deployment | Docker Compose | 3 services: `lumos-qdrant`, `lumos-api`, `lumos-ui` |
| Config | Dataclass + env vars | All tunables via `QDRANT_URL`, `MODEL_NAME`, `DEVICE`, etc. |

---

## Benchmark (Flickr8k test split)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.294 |
| Recall@5 | 0.533 |
| Recall@10 | 0.631 |
| MRR@10 | 0.397 |
| nDCG@10 | 0.452 |

Latency on CPU (Docker, after model warm-up):

| Stage | p50 | p95 |
|-------|-----|-----|
| Encode | ~9 ms | ~11 ms |
| Search | ~24 ms | ~39 ms |
| End-to-End | ~33 ms | ~48 ms |

> First request cold-starts the OpenCLIP model (~1–2 min). Subsequent requests hit the warm path.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Flickr8k images placed in `data/flickr8k/images/` (the compose file bind-mounts this directory)
- The Qdrant collection must be indexed first via the notebook (see [Data Pipeline](#data-pipeline))

### 1. Build & Launch

```bash
docker compose up -d --build
```

This spins up 3 containers:

| Container | Port | URL |
|-----------|------|-----|
| `lumos-qdrant` | 6334 | http://localhost:6334/dashboard |
| `lumos-api` | 8008 | http://localhost:8008/health |
| `lumos-ui` | 8508 | http://localhost:8508 |

### 2. Verify Everything Is Running

Check container status:

```bash
docker compose ps
```

You should see all 3 services as `running`. Then hit the health endpoint:

```bash
# Linux / macOS / Git Bash
curl http://localhost:8008/health

# PowerShell
Invoke-RestMethod http://localhost:8008/health | ConvertTo-Json
```

Expected response:

```json
{
  "status": "ok",
  "model": { "name": "ViT-B-32", "pretrained": "openai" },
  "device": "cpu",
  "qdrant_url": "http://lumos-qdrant:6333",
  "collection": "flickr8k_images"
}
```

> **Note:** The first request takes ~1–2 minutes because the OpenCLIP model needs to download/load into memory. Every request after that is fast.

### 3. Check Qdrant Has Data

Open http://localhost:6334/dashboard in your browser — you should see the `flickr8k_images` collection with ~8,000 indexed vectors. If the collection is empty or missing, you need to run the indexing notebook first (see [Data Pipeline](#data-pipeline)).

---

## Testing the System

### Option A: Streamlit UI (easiest)

Open http://localhost:8508 in your browser. The UI has 3 tabs:

**Tab 1 — Text → Image:**
1. Type a natural language query like `a dog running on the grass`
2. Set `top_k` (default 5)
3. Click **Search (text)**
4. You get a grid of the top matching images with similarity scores and captions

**Example queries to try:**
- `a child playing in the snow`
- `two people sitting on a bench`
- `a man surfing on a wave`
- `sunset over the ocean`
- `a cat sleeping on a couch`

**Tab 2 — Image → Image (upload):**
1. Upload any `.jpg`/`.png`/`.webp` image
2. Click **Search (image upload)**
3. Returns visually similar images from the index — useful for "find more like this"

**Tab 3 — Similar by image_id:**
1. Enter an `image_id` (integer, 0-based — try `0`, `42`, `100`, etc.)
2. Click **Search similar**
3. Returns the nearest images to that indexed image's vector

### Option B: API via curl / PowerShell

#### Text → Image Search

```bash
# Bash
curl -s -X POST http://localhost:8008/search_text \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog running on the grass", "top_k": 3}' | python -m json.tool
```

```powershell
# PowerShell
$body = @{ query = "a dog running on the grass"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8008/search_text -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

Expected output — a ranked list of matches:

```json
{
  "query": "a dog running on the grass",
  "top_k": 3,
  "results": [
    {
      "image_id": 1234,
      "filename": "2345678901.jpg",
      "score": 0.312,
      "split": "train",
      "caption0": "A brown dog runs across a grassy field.",
      "image_url": "/image/2345678901.jpg"
    },
    ...
  ]
}
```

- **`score`** = cosine similarity (0–1, higher = more relevant)
- **`image_url`** = relative path — open `http://localhost:8008/image/2345678901.jpg` in browser to view the actual image
- **`caption0`** = first caption from Flickr8k ground truth

#### Image → Image Search (upload a file)

```bash
# Bash — upload any image file
curl -s -X POST "http://localhost:8008/search_image?top_k=5" \
  -F "file=@./data/flickr8k/images/667626_18933d713e.jpg" | python -m json.tool
```

```powershell
# PowerShell
$filePath = ".\data\flickr8k\images\667626_18933d713e.jpg"
Invoke-RestMethod -Method Post -Uri "http://localhost:8008/search_image?top_k=5" -Form @{ file = Get-Item $filePath }
```

This encodes the uploaded image with CLIP and returns the most visually similar images in the index.

#### Find Similar by image_id

```bash
curl -s "http://localhost:8008/similar_image/42?top_k=5" | python -m json.tool
```

```powershell
Invoke-RestMethod "http://localhost:8008/similar_image/42?top_k=5" | ConvertTo-Json -Depth 5
```

This fetches the stored vector for `image_id=42` from Qdrant and returns its nearest neighbours — no encoding needed, so it's faster.

#### View an Image

Open in browser or download:

```bash
curl -o result.jpg http://localhost:8008/image/667626_18933d713e.jpg
```

```powershell
Invoke-WebRequest -Uri http://localhost:8008/image/667626_18933d713e.jpg -OutFile result.jpg
```

### Option C: FastAPI Swagger Docs

Open http://localhost:8008/docs — FastAPI auto-generates interactive Swagger UI. You can test every endpoint directly from the browser with form inputs and see the raw request/response.

---

## Running Without Docker (Local Dev)

If you want to run the API and UI directly on your machine:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant separately (still needs Docker, or use Qdrant Cloud)
docker run -d -p 6334:6333 --name lumos-qdrant qdrant/qdrant:latest

# 3. Set env vars (PowerShell)
$env:QDRANT_URL = "http://localhost:6334"
$env:QDRANT_COLLECTION = "flickr8k_images"
$env:IMAGES_DIR = "./data/flickr8k/images"
$env:MODEL_NAME = "ViT-B-32"
$env:DEVICE = "cpu"

# 3b. Set env vars (Bash)
export QDRANT_URL=http://localhost:6334
export QDRANT_COLLECTION=flickr8k_images
export IMAGES_DIR=./data/flickr8k/images
export MODEL_NAME=ViT-B-32
export DEVICE=cpu

# 4. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8008

# 5. In a separate terminal — start the UI
$env:API_URL = "http://localhost:8008"   # PowerShell
streamlit run src/ui/app.py --server.port 8508
```

All config is in `src/config.py` — every setting reads from env vars with sensible defaults.

---

## Data Pipeline

The indexing and evaluation pipeline lives in `notebooks/01_flickr8k_end2end.ipynb`. To reproduce:

1. Download the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract images into `data/flickr8k/images/`.
2. Run the notebook — it handles dataset prep, CLIP embedding generation, Qdrant collection creation + upsert, retrieval evaluation, latency benchmarking, and artifact export.
3. The notebook writes parquet artifacts to `artifacts/flickr8k/` which the eval CLI (`src/eval/run_eval.py`) can consume for standalone evaluation runs.

---

## Project Structure

```
lumos/
├── src/
│   ├── config.py              # Settings dataclass, all env-var tunables
│   ├── embeddings/
│   │   └── clip_encoder.py    # OpenCLIP wrapper (text + image encoding)
│   ├── qdrant/
│   │   ├── client.py          # REST client (search, upsert, get_points)
│   │   ├── index.py           # Collection creation + batch upsert
│   │   └── schema.py          # Distance metric + payload field defs
│   ├── search/
│   │   ├── text_to_image.py   # encode text → Qdrant search
│   │   └── image_to_image.py  # encode image / lookup by ID → search
│   ├── eval/
│   │   ├── metrics.py         # Recall@K, MRR@K, nDCG@K
│   │   └── run_eval.py        # CLI evaluation against test split
│   ├── datasets/
│   │   └── flickr8k.py        # Parquet loaders for meta + captions
│   ├── api/
│   │   └── main.py            # FastAPI app
│   └── ui/
│       └── app.py             # Streamlit app
│
├── notebooks/
│   └── 01_flickr8k_end2end.ipynb
│
├── docker/
│   ├── api/Dockerfile
│   └── ui/Dockerfile
│
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---


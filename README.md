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

### Docker Compose (recommended)

```bash
docker compose up -d --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | `http://localhost:8508` |
| FastAPI (health) | `http://localhost:8008/health` |
| Qdrant Dashboard | `http://localhost:6334/dashboard` |

Images must be mounted at `./data/flickr8k/images/` — the compose file bind-mounts this into the API container as read-only.

### Local (no Docker)

```bash
pip install -r requirements.txt
# start Qdrant separately (e.g. via Docker or Qdrant Cloud)
uvicorn src.api.main:app --port 8008
streamlit run src/ui/app.py --server.port 8508
```

Set env vars to override defaults (see `src/config.py`):

```bash
export QDRANT_URL=http://localhost:6334
export QDRANT_COLLECTION=flickr8k_images
export IMAGES_DIR=./data/flickr8k/images
export MODEL_NAME=ViT-B-32
export DEVICE=cpu
```

---

## API

### `GET /health`

```bash
curl http://localhost:8008/health
```

### `POST /search_text`

```bash
curl -X POST http://localhost:8008/search_text \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog running on the grass", "top_k": 5}'
```

### `POST /search_image`

```bash
curl -X POST http://localhost:8008/search_image?top_k=6 \
  -F "file=@photo.jpg"
```

### `GET /similar_image/{image_id}`

```bash
curl "http://localhost:8008/similar_image/42?top_k=6"
```

### `GET /image/{filename}`

Proxies raw image bytes from the mounted images directory.

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

## License

MIT — see [LICENSE](LICENSE).

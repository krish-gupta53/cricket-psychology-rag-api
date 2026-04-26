# Cricket Psychology RAG API

A FastAPI service that generates cricket psychology social media content using a 4-stage RAG pipeline with Pinecone and OpenAI.

## Pipeline

1. **Node 1** — Generate structured content brief from the input fields
2. **Node 2** — Generate 3 Pinecone retrieval queries (symptom, mental skill, deep psychology)
3. **Pinecone** — Retrieve 5 chunks per query (15 total), select best 9 using diversity-aware reranking
4. **Node 3** — Generate 5 content angles, select the strongest one
5. **Node 4** — Generate complete content pack (reel script, carousel, caption, hashtags, mental drill)

## Models

- **LLM**: `gpt-4o-mini` for all 4 nodes
- **Embeddings**: `text-embedding-3-small` for Pinecone queries

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/generate-content` | Run full pipeline |
| GET | `/docs` | Swagger UI |

## Request Body

```json
{
  "problem": "feel fear when going to chase a big total as an opener",
  "category": "batting psychology",
  "sub_category": "chasing pressure",
  "cricket_role": "opening batter",
  "player_level": "club",
  "psychology_angle": "fear of failure",
  "debug": true
}
```

Set `debug: true` while testing to also return content brief, queries, selected chunks, and angle selection in the response. Set `debug: false` for production clients.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Fill in your real API keys
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test via Swagger UI.

## Deploy on Render (Recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` — or manually set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Go to **Environment** and add all variables from `.env.example`
6. Click **Deploy**
7. Test `https://your-app.onrender.com/health`

## Deploy on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) and create a new project from your repo
3. Add all environment variables from `.env.example`
4. Deploy the service
5. Go to **Networking** → **Generate Domain** to get a public URL
6. Test `/health` and `/docs`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `PINECONE_API_KEY` | Yes | — | Pinecone API key |
| `PINECONE_INDEX_HOST` | Yes | — | Full Pinecone index host URL |
| `PINECONE_NAMESPACE` | No | `content-automate` | Pinecone namespace |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `OPENAI_LLM_MODEL` | No | `gpt-4o-mini` | LLM for all 4 nodes |
| `TOP_K_PER_QUERY` | No | `5` | Pinecone results per query |
| `FINAL_CHUNK_COUNT` | No | `9` | Final chunks sent to content nodes |

## Top-9 Chunk Selection Logic

- Deduplicate 15 candidates by chunk ID and text hash
- Normalize scores within each query type group
- Guarantee at least 2 chunks from each query type (symptom, mental_skill, deep_psychology)
- Fill remaining slots by highest combined score (relevance + diversity bonus + source priority)

## Notes

- `debug=true` adds significant response size — use `false` in production API clients
- If any LLM node returns malformed JSON, the endpoint returns a `502` error with details
- Keep `TOP_K_PER_QUERY=5` on free tier to stay within Pinecone read unit limits

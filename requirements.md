### Project requirements (notebook-first)

- **Goal**: Build a semantic book recommender using embeddings + vector DB, zero-shot category fill, emotion-based sorting, and a simple dashboard. All steps will be developed in Jupyter notebooks.

### Prerequisites
- **Python**: 3.10–3.12 (prefer 3.11)
- **OS**: macOS (Apple Silicon supported). CPU-only is fine; GPU optional.
- **Accounts/Keys**:
  - **Kaggle**: access to the “7k books” dataset
  - **OpenAI API key** (optional, default for embeddings) or choose a local HF embedding model
  - **Hugging Face token** (optional, for rate-limited models)
- **Disk**: ~2–4 GB for data, models, and vector index
- **Budget**: If using OpenAI embeddings, a few dollars should suffice

### Environment & dependencies
- **Virtual env**: venv or conda
- **Environment variables** (via `.env` + `python-dotenv`):
  - `OPENAI_API_KEY` (if using OpenAI)
  - `HUGGINGFACEHUB_API_TOKEN` (optional)
- **Core packages**:
  - Data: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
  - LLM tooling: `langchain`, `langchain-community`, `chromadb`, `openai` (or `langchain-openai`), `transformers`
  - Dashboard: `gradio`
  - Utils: `python-dotenv`, `kagglehub`

### Data assets
- Source: Kaggle 7k books (titles, authors, categories, description, ratings, cover URL)
- Outputs we will create and reuse:
  - `books_cleaned.csv` (filtered/engineered features)
  - `tag_descriptions.txt` (ISBN + description, one per line)
  - `chroma_index/` (persisted vector DB for fast startup)

### Notebook workflow (sequenced)
- `01_data_exploration_cleaning.ipynb`:
  - Load raw dataset; inspect; handle missing; enforce min description length; create `title_and_subtitle`, `tag_description`
  - Save `books_cleaned.csv`
- `02_vector_search.ipynb`:
  - Choose embeddings (OpenAI or local HF); create embeddings; build Chroma; persist to `chroma_index/`
  - Similarity search; map back to books via ISBN
- `03_zero_shot_category.ipynb`:
  - Collapse categories → {fiction, non-fiction, children’s fiction, children’s non-fiction}
  - Use zero-shot (BART MNLI) to fill missing; validate on labeled subset
- `04_sentiment_emotions.ipynb`:
  - Sentence-level emotion with a fine-tuned RoBERTa; aggregate max per emotion per book
  - Merge scores into dataset; save updated CSV
- `05_dashboard_gradio.ipynb`:
  - Load persisted Chroma and datasets; implement query + filters (category, tone); display covers + captions

### Cover images
- Use higher-res Google Books thumbnails when available; fallback to `cover-not-found.jpg`

### Decisions to confirm
- Embedding provider default: OpenAI (cheapest/fastest to start) vs local HF model
- Minimum description length threshold (default: ≥25 words)
- Persisted index path and naming (`chroma_index/`)

### Next steps
1) Set up virtual env and install packages
2) Create `.env` and add any keys
3) Start `01_data_exploration_cleaning.ipynb`


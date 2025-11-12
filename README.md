# Semantic Book Recommender

An intelligent book recommendation system powered by large language models, vector embeddings, and sentiment analysis. This project demonstrates end-to-end development of a production-ready recommendation engine that understands semantic meaning, categorizes content intelligently, and filters by emotional tone.

## Overview

This semantic book recommender goes beyond traditional keyword matching by leveraging transformer-based embeddings to understand the true meaning and context of book descriptions. Users can search using natural language queries, filter by category, and sort by emotional tone to discover books that match their mood and interests.

## Project Architecture

The system processes book data through a sequential pipeline, transforming raw data into a searchable, categorized, and emotion-annotated dataset.

**Data Pipeline Flow:**
1. Raw dataset (7k books from Kaggle) → Data cleaning and feature engineering
2. Cleaned data → Vector embeddings generation → ChromaDB persistence
3. Cleaned data → Zero-shot category classification → Category assignment
4. Cleaned data → Sentence-level emotion analysis → Emotion scores
5. Final dataset + Vector DB → Streamlit dashboard for user interaction

**Technical Implementation Details:**

**Embedding Generation:**
- Uses OpenAI's `text-embedding-3-small` model via LangChain
- Processes descriptions with ISBN prefixes for vector-to-book mapping
- Implements token-based batching (250K tokens/batch) to handle OpenAI's 300K token request limit
- Stores embeddings in ChromaDB with persistent index in `data/chroma_index/`

**Category Classification:**
- Maps 500+ raw categories to 4 simplified categories: Fiction, Nonfiction, Children's Fiction, Children's Nonfiction
- Uses Hugging Face BART-large-MNLI for zero-shot classification
- Validates on 300 labeled samples achieving 75-80% accuracy
- Fills missing categories for books without manual labels

**Emotion Analysis:**
- Splits book descriptions into sentences for granular analysis
- Processes each sentence through fine-tuned RoBERTa emotion classifier
- Extracts maximum probability per emotion across all sentences (7 emotions: anger, disgust, fear, joy, sadness, surprise, neutral)
- Aggregates to single max score per emotion per book

**Search & Retrieval:**
- User query → OpenAI embedding → Cosine similarity search in ChromaDB
- Retrieves top 50 semantically similar books
- Filters by category if specified
- Sorts by emotion scores if tone selected
- Returns top 16 results to dashboard

## Data Flow & Processing

**Input:** Raw CSV with 7,000+ books from Kaggle dataset

**Processing Steps:**
1. **Data Cleaning:** Removes books with missing descriptions, filters descriptions < 25 words, handles missing subtitles
2. **Feature Engineering:** Creates `tag_description` column (ISBN + description) for vector search linking
3. **Embedding Generation:** Converts all descriptions to 1536-dimensional vectors using OpenAI embeddings
4. **Category Assignment:** Maps known categories, classifies unknown using zero-shot
5. **Emotion Scoring:** Processes 5,197 books through emotion classifier, extracts max scores
6. **Output:** Final dataset with categories, emotions, and vector index ready for search

**Vector Search Query Flow:**
```
User Query ("book about forgiveness")
  → Embedding generation (OpenAI API)
  → Similarity search (ChromaDB, cosine similarity)
  → Top 50 results with scores
  → ISBN extraction from document text
  → DataFrame lookup by ISBN
  → Category filter (if specified)
  → Emotion sort (if tone selected)
  → Top 16 results returned
```

**Performance Characteristics:**
- Vector search: ~200-500ms (embedding + similarity computation)
- DataFrame filtering: < 50ms
- Emotion sorting: < 10ms
- Total response time: < 1 second for typical queries

## System Architecture

**Data Processing Pipeline:**

The project follows a notebook-based ETL workflow where each phase outputs artifacts used by subsequent phases:

```
Raw Data (Kaggle) 
  ↓ [01_data_exploration_cleaning.ipynb]
  books_cleaned.csv (5,197 books, 25+ word descriptions)
  ↓
  ├─→ [02_vector_search.ipynb] → tag_description.txt → ChromaDB embeddings → data/chroma_index/
  ├─→ [03_text_classification.ipynb] → books_with_categories.csv (all categories assigned)
  └─→ [04_sentiment_analysis.ipynb] → books_with_emotions.csv (7 emotion columns)
```

**Vector Database Structure:**
- Each document stored as: `"{ISBN} {description}"`
- Enables ISBN extraction from search results for book lookup
- Persisted to disk to avoid re-embedding on startup
- Load time: < 2 seconds for 272 document embeddings

**Category Mapping Logic:**
- 12 major categories manually mapped (e.g., "Juvenile Fiction" → "Children's Fiction")
- Remaining books classified via zero-shot with candidate labels: ["Fiction", "Nonfiction"]
- Missing categories filled automatically using description text

**Emotion Processing Workflow:**
- Description split by periods into sentences
- Batch processing: all sentences from one book classified together
- Score aggregation: max probability per emotion across sentences
- Result: 7 float columns (0-1 range) per book

**Dashboard Implementation:**
- Loads pre-computed data on startup (cached with `@st.cache_data` and `@st.cache_resource`)
- Vector DB loaded once per session (not rebuilt)
- Search function: `similarity_search()` → ISBN extraction → DataFrame filtering → Emotion sorting
- UI: 4-column responsive grid with book covers, titles, authors, truncated descriptions

## Technical Decisions & Trade-offs

**Embedding Model Choice:**
OpenAI's `text-embedding-3-small` selected over local models for accuracy and API simplicity. Trade-off: requires API key and incurs costs (~$0.10 per 1K searches).

**Vector Database:**
ChromaDB chosen for persistence and LangChain integration. Alternative vector stores (Pinecone, Weaviate) would require cloud infrastructure.

**Token Batching:**
Implemented custom token counting using `tiktoken` to batch embeddings within OpenAI's 300K token/request limit. Processes 250K tokens per batch to leave buffer.

**Category Simplification:**
Reduced 500+ categories to 4 simplified categories for better filtering UX. Zero-shot classification handles generalization without training data.

**Emotion Granularity:**
Sentence-level processing chosen over whole-description to capture emotional variety. Max aggregation ensures strongest emotion per book is captured.

**Data Format:**
CSV files used for simplicity and version control. For production, consider SQLite or PostgreSQL for better query performance.

**Caching Strategy:**
Streamlit's `@st.cache_data` for CSV loads, `@st.cache_resource` for vector DB. Prevents reloading on every user interaction.

## Technical Stack

- **Languages**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Transformers (Hugging Face), OpenAI API
- **Vector Database**: ChromaDB with LangChain integration
- **NLP Models**: 
  - BART-large-MNLI for zero-shot classification
  - RoBERTa-based emotion classifier (j-hartmann/emotion-english-distilroberta-base)
- **Web Framework**: Streamlit
- **Data Visualization**: Matplotlib, Seaborn
- **Environment Management**: python-dotenv

## Project Structure

```
llm-semantic-book-recommender/
├── notebooks/
│   ├── 01_data_exploration_cleaning.ipynb
│   ├── 02_vector_search.ipynb
│   ├── 03_text_classification.ipynb
│   └── 04_sentiment_analysis.ipynb
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_categories.csv
│   ├── books_with_emotions.csv
│   └── chroma_index/ (persisted vector database)
├── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

## Getting Started

1. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API keys**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. **Run notebooks sequentially**
   Execute notebooks 01-04 in order to generate all required data files.

4. **Rebuild the Chroma index locally**
   ```bash
   source venv/bin/activate  # if not already active
   python scripts/rebuild_chroma.py
   ```

5. **Launch dashboard**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

## Performance Metrics

- **Dataset Size**: 5,197 books after cleaning and filtering
- **Vector Database**: 272 embeddings stored in persistent ChromaDB
- **Category Classification**: 75-80% accuracy on validation set
- **Emotion Analysis**: Sentence-level processing with 7 emotion dimensions
- **Search Response Time**: Sub-second for semantic similarity queries

## Future Enhancements

Potential improvements include multi-modal embeddings that incorporate book cover images, collaborative filtering based on user ratings, and advanced emotion modeling that captures emotional arcs throughout books. The architecture supports easy integration of additional features such as author-based recommendations, reading time estimation, and personalized ranking algorithms.

## License

This project is developed for learning and portfolio purposes. Dataset sourced from Kaggle's "7k books" dataset.


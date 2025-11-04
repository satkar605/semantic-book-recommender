# Work Plan: Semantic Book Recommender Project

## Overview
Build an end-to-end semantic book recommender using LLMs, vector search, text classification, and sentiment analysis. All development will be done in Jupyter notebooks for interactive exploration and learning.

---

## Phase 0: Project Setup & Environment

### Tasks
1. **Create virtual environment**
   - Create venv/conda environment (Python 3.10-3.12)
   - Activate environment

2. **Install dependencies**
   - Install core packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`
   - Install LLM packages: `langchain`, `langchain-community`, `langchain-openai`, `chromadb`, `openai`, `transformers`
   - Install dashboard: `gradio`
   - Install utils: `python-dotenv`, `kagglehub`
   - Install Jupyter: `jupyter`, `ipykernel`, `ipywidgets`

3. **Set up environment variables**
   - Create `.env` file in project root
   - Add `OPENAI_API_KEY` (if using OpenAI embeddings)
   - Add `HUGGINGFACEHUB_API_TOKEN` (optional, for rate-limited models)
   - Create `.env.example` template (without actual keys)

4. **Project structure**
   - Create `data/` directory for raw/processed data
   - Create `notebooks/` directory for all Jupyter notebooks
   - Create `chroma_index/` directory for vector database persistence
   - Verify `cover-not-found.jpg` is accessible

### Deliverables
- ✅ Working virtual environment
- ✅ All packages installed and importable
- ✅ `.env` file configured
- ✅ Project directory structure created

### Success Criteria
- Can import all required packages without errors
- Environment variables load correctly
- Project folders are ready

---

## Phase 1: Data Exploration & Cleaning

### Notebook: `01_data_exploration_cleaning.ipynb`

### Tasks
1. **Load and inspect raw data**
   - Use `kagglehub` to download "7k books" dataset
   - Read into pandas DataFrame
   - Display basic info: shape, columns, dtypes
   - Use PyCharm's column statistics feature (if available) or manual exploration

2. **Analyze missing values**
   - Create missingness heatmap using seaborn
   - Identify patterns in missing data
   - Check for bias: correlate missing descriptions with book age, pages, ratings

3. **Data quality checks**
   - Check for duplicate ISBNs
   - Analyze category distribution (too many specific categories)
   - Check description quality (word count distribution)

4. **Data cleaning**
   - Remove books with missing descriptions, pages, ratings, or published year
   - Filter descriptions with <25 words (too short to be meaningful)
   - Create `title_and_subtitle` field (combine with colon, handle missing)
   - Create `tag_description` field (ISBN + description for vector DB linking)

5. **Save cleaned dataset**
   - Export to `data/books_cleaned.csv`
   - Verify output: check shape, no missing descriptions, all required fields present

### Deliverables
- ✅ Cleaned dataset: `data/books_cleaned.csv`
- ✅ Documentation of cleaning decisions (in notebook markdown cells)
- ✅ Summary statistics: final book count, category distribution

### Success Criteria
- Dataset has ~5,000+ books (after filtering)
- No missing values in critical fields (description, ISBN)
- All descriptions have ≥25 words
- `tag_description` field properly formatted (ISBN + description)

---

## Phase 2: Vector Search & Embeddings

### Notebook: `02_vector_search.ipynb`

### Tasks
1. **Set up embedding provider**
   - Load OpenAI API key from `.env`
   - Initialize OpenAI embeddings model (or choose local HF alternative)
   - Test embedding generation on sample text

2. **Prepare text for vectorization**
   - Load `books_cleaned.csv`
   - Extract `tag_description` column
   - Save to `data/tag_descriptions.txt` (one per line, no header/index)

3. **Build vector database**
   - Use LangChain `TextLoader` to load descriptions
   - Use `CharacterTextSplitter` (chunk_size=0, separator='\n', chunk_overlap=0)
   - Create embeddings using OpenAI embeddings
   - Build Chroma vector database from documents
   - **Persist to disk**: Save Chroma DB to `chroma_index/` directory

4. **Test similarity search**
   - Create test query: "a book to teach children about nature"
   - Run `similarity_search` with k=10
   - Extract ISBNs from returned descriptions
   - Map back to full book data using ISBN matching
   - Display results (title, author, description snippet)

5. **Create reusable function**
   - Function: `retrieve_semantic_recommendations(query, top_k=10)`
   - Returns pandas DataFrame with book recommendations
   - Test with multiple queries

### Deliverables
- ✅ Persisted Chroma vector database in `chroma_index/`
- ✅ `data/tag_descriptions.txt` file
- ✅ Working `retrieve_semantic_recommendations()` function
- ✅ Test results showing relevant book matches

### Success Criteria
- Vector DB loads from disk quickly (no rebuild needed)
- Semantic search returns relevant books for test queries
- ISBN extraction and mapping works correctly
- Function handles edge cases (empty results, etc.)

---

## Phase 3: Zero-Shot Category Classification

### Notebook: `03_zero_shot_category.ipynb`

### Tasks
1. **Load and prepare category data**
   - Load `books_cleaned.csv`
   - Analyze existing category distribution
   - Identify top categories (fiction, non-fiction, juvenile fiction, etc.)

2. **Create category mapping**
   - Map top 12 categories to simplified categories:
     - Fiction, Non-fiction
     - Children's Fiction, Children's Non-fiction
   - Create `simple_category` column with known labels
   - Identify books with missing categories (need prediction)

3. **Set up zero-shot classifier**
   - Use Hugging Face model selector (BART-large-MNLI or similar)
   - Load model using `transformers` pipeline
   - Configure device (CPU/MPS/CUDA based on hardware)

4. **Validate classifier accuracy**
   - Sample 300 fiction + 300 non-fiction books with known labels
   - Run predictions on sample
   - Calculate accuracy (should be ~75-80%)
   - Create confusion matrix if helpful

5. **Classify missing categories**
   - Extract descriptions for books with missing `simple_category`
   - Run zero-shot classification (batch processing with progress bar)
   - Store predictions in DataFrame
   - Merge predictions back into main dataset

6. **Save updated dataset**
   - Verify all books now have categories
   - Save to `data/books_with_categories.csv`

### Deliverables
- ✅ Updated dataset: `data/books_with_categories.csv`
- ✅ All books have `simple_category` assigned
- ✅ Validation accuracy metrics documented

### Success Criteria
- Zero-shot classifier achieves >75% accuracy on known labels
- All books have category assigned (no missing values)
- Category distribution is reasonable (not all one category)

---

## Phase 4: Sentiment & Emotion Analysis

### Notebook: `04_sentiment_emotions.ipynb`

### Tasks
1. **Load data and set up emotion classifier**
   - Load `books_with_categories.csv`
   - Find fine-tuned emotion model on Hugging Face (RoBERTa-based, 6-7 emotions)
   - Load model: `j-hartmann/emotion-english-roberta-large` or similar
   - Configure for 7 emotions: anger, disgust, fear, joy, sadness, surprise, neutral

2. **Test emotion classification approach**
   - Test on full description vs. sentence-level
   - Decide: sentence-level gives better granularity
   - Verify sentence splitting works correctly

3. **Process all books**
   - Loop through each book description:
     - Split into sentences
     - Classify each sentence
     - Extract max probability per emotion across all sentences
   - Store results: ISBN + 7 emotion scores

4. **Create emotion DataFrame**
   - Create DataFrame with columns: ISBN, anger, disgust, fear, joy, sadness, surprise, neutral
   - Merge with main books DataFrame
   - Verify emotion scores are reasonable (0-1 range)

5. **Analyze emotion distribution**
   - Visualize distribution of emotions
   - Identify which emotions are most common
   - Save updated dataset

6. **Save final dataset**
   - Save to `data/books_final.csv` (with categories + emotions)

### Deliverables
- ✅ Final dataset: `data/books_final.csv`
- ✅ Emotion scores for all books (7 emotion columns)
- ✅ Emotion distribution analysis

### Success Criteria
- All books have emotion scores
- Emotion scores are in valid range (0-1)
- Distribution shows variety (not all neutral/joy)
- Processing completes without errors

---

## Phase 5: Gradio Dashboard

### Notebook: `05_dashboard_gradio.ipynb`

### Tasks
1. **Load all components**
   - Load persisted Chroma vector database
   - Load `books_final.csv`
   - Verify cover image fallback (`cover-not-found.jpg`) is accessible

2. **Prepare cover images**
   - Modify thumbnail URLs to get higher resolution (add `&edge=curl` parameter)
   - Replace missing thumbnails with `cover-not-found.jpg`

3. **Create recommendation function**
   - Function: `recommend_books(query, category='all', tone='all', top_k=16)`
   - Integrate semantic search
   - Apply category filter (fiction/non-fiction/children's)
   - Apply tone sorting (happy/surprising/angry/suspenseful/sad)
   - Map tone labels to emotion columns (happy→joy, suspenseful→fear, etc.)

4. **Format book display**
   - Create caption format: "Title by Author: [truncated description]"
   - Truncate descriptions to 30 words max
   - Format author names (handle multiple authors with "and")
   - Return list of tuples: (thumbnail_url, caption)

5. **Build Gradio interface**
   - Create `Blocks` with theme (suggest: "glass" or "soft")
   - Add title markdown: "Semantic Book Recommender"
   - Add input components:
     - Textbox for query (with placeholder)
     - Dropdown for category (all/fiction/non-fiction/children's fiction/children's non-fiction)
     - Dropdown for tone (all/happy/surprising/angry/suspenseful/sad)
     - Submit button
   - Add output: Gallery component (columns=8, rows=2, max 16 results)
   - Wire up `recommend_books()` function to interface

6. **Test dashboard**
   - Test with various queries
   - Test category filtering
   - Test tone sorting
   - Verify images load correctly
   - Check responsive behavior

7. **Launch and demo**
   - Launch dashboard
   - Test with example queries:
     - "a book about World War I"
     - "a story about a troubled family across generations"
   - Verify recommendations are relevant

### Deliverables
- ✅ Working Gradio dashboard
- ✅ All filters and sorting functional
- ✅ Book covers display correctly
- ✅ Recommendations are relevant and diverse

### Success Criteria
- Dashboard launches without errors
- Semantic search returns relevant books
- Category filter works correctly
- Tone sorting shows books with higher emotion scores first
- Images load (either cover or fallback)
- UI is clean and user-friendly

---

## Phase 6: Testing & Refinement

### Tasks
1. **End-to-end testing**
   - Test full pipeline from query to results
   - Verify all data files are properly loaded
   - Check for any edge cases (empty queries, no results, etc.)

2. **Performance optimization**
   - Ensure Chroma DB loads quickly (not rebuilt each time)
   - Optimize emotion classification if needed (batch processing)
   - Add caching if dashboard is slow

3. **Documentation**
   - Add markdown cells explaining each step
   - Document any assumptions or decisions
   - Add usage instructions for dashboard

4. **Optional enhancements**
   - Add more filters (by rating, year, pages)
   - Improve cover image handling
   - Add export functionality for recommendations
   - Enhance UI styling

---

## Dependencies & Order

```
Phase 0 (Setup) → Phase 1 (Data Cleaning) → Phase 2 (Vector Search)
                                                     ↓
Phase 3 (Categories) ← Phase 2 (Vector Search) → Phase 4 (Emotions)
                                                     ↓
Phase 5 (Dashboard) ← Phase 2 + Phase 3 + Phase 4
```

**Critical Path:**
- Must complete Phase 1 before Phase 2 (need cleaned data)
- Must complete Phase 2 before Phase 5 (need vector DB)
- Phase 3 and Phase 4 can be done in parallel (both need Phase 1, independent)
- Phase 5 requires Phase 2, 3, and 4 complete

---

## Estimated Timeline

- **Phase 0**: 30-60 minutes (setup, installation)
- **Phase 1**: 2-3 hours (data exploration, cleaning)
- **Phase 2**: 2-3 hours (embeddings, vector DB, testing)
- **Phase 3**: 2-3 hours (classification, validation)
- **Phase 4**: 3-4 hours (emotion processing, batch work)
- **Phase 5**: 2-3 hours (dashboard building, testing)
- **Phase 6**: 1-2 hours (testing, refinement)

**Total**: ~12-18 hours of focused work

---

## Notes & Tips

1. **Notebook Best Practices**
   - Save frequently
   - Use markdown cells for documentation
   - Test each cell before moving to next
   - Keep cells focused on one task

2. **API Costs**
   - OpenAI embeddings: ~$0.0001 per 1K tokens (very cheap)
   - $10 should cover extensive testing
   - Monitor usage in OpenAI dashboard

3. **Performance**
   - Emotion classification is slowest step (5000+ books)
   - Use `tqdm` for progress bars
   - Consider batching if needed

4. **Debugging**
   - Check data types (ISBN as string vs int)
   - Verify API keys are loaded
   - Test with small samples first

5. **Learning Opportunities**
   - Experiment with different embedding models
   - Try different emotion models
   - Adjust category thresholds
   - Customize dashboard theme

---

## Success Metrics

- ✅ All notebooks complete and runnable
- ✅ Dashboard returns relevant recommendations
- ✅ Code is well-documented and understandable
- ✅ All data artifacts are properly saved
- ✅ Project can be reproduced from scratch

---

## Ready to Start?

Begin with **Phase 0: Project Setup** and we'll work through each phase together. Let me know when you're ready to start coding!


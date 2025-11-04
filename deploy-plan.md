# Deployment Plan: Semantic Book Recommender

## Overview

This document outlines the deployment process for the Semantic Book Recommender Streamlit dashboard. The application can be deployed to Streamlit Community Cloud (recommended) or alternative platforms.

## Pre-Deployment Checklist

### Required Files
- [x] `streamlit_dashboard.py` - Main dashboard application
- [x] `requirements.txt` - Python dependencies
- [x] `data/books_with_emotions.csv` - Complete dataset with all books (verify it has 5,197+ rows)
- [x] `data/chroma_index/` - Persisted vector database directory
- [x] `cover-not-found.jpg` - Fallback image for missing book covers
- [x] `.gitignore` - Excludes `.env` and sensitive files

### Data Verification
1. **Verify `books_with_emotions.csv` is complete:**
   ```bash
   wc -l data/books_with_emotions.csv
   # Should show 5,199 lines (5,197 books + header + empty line)
   ```

2. **Verify ChromaDB index exists:**
   ```bash
   ls -la data/chroma_index/
   # Should contain chroma.sqlite3 and subdirectories
   ```

3. **Test locally before deployment:**
   ```bash
   source venv/bin/activate
   streamlit run streamlit_dashboard.py
   # Verify search works and returns results
   ```

### Code Review
- [x] All emojis removed from dashboard
- [x] Error handling in place
- [x] Caching implemented for performance
- [x] Path references are relative (not absolute)
- [x] No hardcoded API keys in code

## Deployment Option 1: Streamlit Community Cloud (Recommended)

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free at share.streamlit.io)
3. OpenAI API key (for embeddings)

### Step-by-Step Process

#### 1. Prepare Repository
```bash
# Initialize git if not already done
git init

# Create .gitignore if missing
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
.env
*.ipynb_checkpoints
.DS_Store
EOF

# Add all files except those in .gitignore
git add .

# Commit changes
git commit -m "Initial commit: Semantic Book Recommender"
```

#### 2. Push to GitHub
```bash
# Create repository on GitHub first, then:
git remote add origin https://github.com/yourusername/semantic-book-recommender.git
git branch -M main
git push -u origin main
```

#### 3. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository
4. Set main file path: `streamlit_dashboard.py`
5. Click "Deploy"

#### 4. Configure Secrets
1. In Streamlit Cloud app settings, go to "Secrets"
2. Add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. Save and restart the app

#### 5. Verify Deployment
- Check that the app loads without errors
- Test a search query (e.g., "a book about forgiveness")
- Verify book covers display correctly
- Check that filters work (category, tone)

### Important Considerations for Streamlit Cloud

**File Size Limits:**
- Total app size: 1GB limit (free tier)
- Individual files: 200MB limit
- `data/chroma_index/` may be large - verify it's under limits

**Data Files:**
- All data files must be in the repository
- Consider using Git LFS for large files if needed:
  ```bash
  git lfs install
  git lfs track "data/chroma_index/**"
  git add .gitattributes
  ```

**Performance:**
- First load may be slow (caching will help subsequent loads)
- Vector DB loading happens once per session
- Consider adding a loading indicator

## Deployment Option 2: Alternative Platforms

### Heroku
**Pros:** Custom domain, more control  
**Cons:** Requires credit card, more complex setup

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Create `setup.sh` for buildpack
4. Set environment variables in Heroku dashboard
5. Deploy: `git push heroku main`

### AWS EC2 / Google Cloud / Azure
**Pros:** Full control, scalable  
**Cons:** More complex, requires infrastructure knowledge

1. Launch instance (Ubuntu recommended)
2. Install Python, dependencies
3. Clone repository
4. Set up environment variables
5. Run with: `streamlit run streamlit_dashboard.py --server.port=8501 --server.address=0.0.0.0`
6. Configure firewall/security groups
7. Set up reverse proxy (nginx) if needed

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t book-recommender .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key book-recommender
```

## Environment Configuration

### Required Environment Variables
- `OPENAI_API_KEY` - For generating embeddings (required)

### Optional Environment Variables
- `HUGGINGFACEHUB_API_TOKEN` - If using HF models (not required for this project)

### Setting Variables

**Streamlit Cloud:**
- Use Secrets tab in app settings

**Local/Other platforms:**
- Create `.env` file (not committed to git)
- Or export: `export OPENAI_API_KEY=your_key`

## File Path Considerations

### Current Paths (Relative)
All paths in `streamlit_dashboard.py` are relative:
- `data/books_with_emotions.csv`
- `data/chroma_index/`
- `cover-not-found.jpg`

These work correctly when:
- Running from project root
- Deployed to Streamlit Cloud (executes from root)
- Running in Docker (WORKDIR set to project root)

### If Issues Arise
If paths don't work, use absolute paths based on script location:
```python
from pathlib import Path
BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "data" / "books_with_emotions.csv"
```

## Post-Deployment Testing

### Functional Tests
1. **Search Functionality:**
   - Test: "a book about personal growth"
   - Expected: Returns relevant nonfiction books
   
2. **Category Filtering:**
   - Test: Search with "Fiction" category selected
   - Expected: Only fiction books returned

3. **Emotion Sorting:**
   - Test: Search with "Happy" tone selected
   - Expected: Books sorted by joy score (highest first)

4. **Edge Cases:**
   - Empty query handling
   - No results scenario
   - Missing image fallback

### Performance Tests
- Initial page load time
- Search response time (should be < 2 seconds)
- Memory usage (check Streamlit Cloud metrics)

### User Experience Tests
- Sidebar instructions display correctly
- Book cards render properly
- Images load (or fallback works)
- Error messages are helpful

## Troubleshooting

### Common Issues

**Issue: "No module named 'langchain_openai'"**
- **Solution:** Ensure `requirements.txt` includes `langchain-openai>=0.3.0`
- Verify installation: `pip install -r requirements.txt`

**Issue: "FileNotFoundError: data/books_with_emotions.csv"**
- **Solution:** Verify file exists and is committed to git
- Check path is relative to script location

**Issue: "No recommendations found"**
- **Solution:** Verify vector database is loaded correctly
- Check ChromaDB index exists in `data/chroma_index/`
- Verify ISBN extraction is working

**Issue: "OPENAI_API_KEY not found"**
- **Solution:** Set secret in Streamlit Cloud Secrets
- Or export environment variable before running

**Issue: App crashes on startup**
- **Solution:** Check Streamlit Cloud logs
- Verify all dependencies are in `requirements.txt`
- Test locally first to catch errors

**Issue: Slow performance**
- **Solution:** Verify caching is working (`@st.cache_data`, `@st.cache_resource`)
- Check vector DB is persisted (not rebuilding)
- Consider reducing `initial_top_k` if too many results

### Debugging on Streamlit Cloud
1. Check app logs in Streamlit Cloud dashboard
2. Add temporary debug output:
   ```python
   st.write(f"Debug: Books loaded: {len(books)}")
   st.write(f"Debug: Vector DB ready: {db_books is not None}")
   ```

## Maintenance

### Regular Updates
- Monitor OpenAI API usage and costs
- Update dependencies periodically
- Refresh dataset if new books are added
- Monitor app performance metrics

### Data Updates
If dataset needs updating:
1. Re-run notebooks 01-04 in sequence
2. Commit updated `books_with_emotions.csv` and `chroma_index/`
3. Push to GitHub
4. Streamlit Cloud will auto-redeploy

### Scaling Considerations
- Current setup handles ~5,000 books efficiently
- For larger datasets, consider:
  - Database instead of CSV
  - Separate embedding service
  - Caching layer (Redis)
  - Load balancing for multiple instances

## Security Best Practices

1. **API Keys:**
   - Never commit `.env` file
   - Use platform secrets management
   - Rotate keys periodically

2. **Data Privacy:**
   - Verify dataset doesn't contain sensitive information
   - Consider data anonymization if needed

3. **Access Control:**
   - Streamlit Cloud: Public by default (can be private with paid plan)
   - Consider authentication if needed

## Cost Estimation

### Streamlit Cloud
- **Free tier:** Unlimited apps, public only
- **Team tier:** $20/month (private apps, custom domains)

### OpenAI API
- **Embeddings:** ~$0.0001 per 1K tokens
- **Estimated:** $0.10-0.50 per 1,000 searches
- **Monthly:** $5-20 for moderate usage

### Total Estimated Cost
- **Free deployment:** $0 (Streamlit) + $5-20/month (OpenAI) = $5-20/month
- **Paid deployment:** $20/month (Streamlit) + $5-20/month (OpenAI) = $25-40/month

## Success Criteria

Deployment is successful when:
- [x] App loads without errors
- [x] Search returns relevant results
- [x] All filters work correctly
- [x] Images display properly
- [x] Performance is acceptable (< 3s response time)
- [x] No sensitive data exposed
- [x] API keys secured
- [x] Documentation complete

## Next Steps After Deployment

1. Share the deployed URL
2. Test with real users
3. Gather feedback
4. Monitor usage and performance
5. Iterate based on feedback
6. Update README with live link
7. Add to portfolio/resume

## Support Resources

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/
- OpenAI API Docs: https://platform.openai.com/docs
- ChromaDB Docs: https://docs.trychroma.com/

---

**Last Updated:** After project completion  
**Status:** Ready for deployment


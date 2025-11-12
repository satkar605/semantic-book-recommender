import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from scripts.rebuild_chroma import main as rebuild_vector_index

load_dotenv()

# Cache data loading for better performance
@st.cache_data
def load_books():
    books = pd.read_csv("data/books_with_emotions.csv")
    books["isbn13"] = books["isbn13"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

@st.cache_resource
def load_vector_db(index_version: float):
    embeddings = OpenAIEmbeddings()
    db_books = Chroma(
        persist_directory="data/chroma_index",
        embedding_function=embeddings,
        collection_name="books"
    )
    return db_books

# Load dataset
books = load_books()
db_books = None

def ensure_vector_index() -> float | None:
    index_path = Path("data/chroma_index/chroma.sqlite3")
    if index_path.exists():
        return index_path.stat().st_mtime

    with st.spinner("Vector index missing. Building it now..."):
        try:
            rebuild_vector_index()
        except Exception as exc:
            st.error(
                "Unable to build the vector index automatically. "
                "Check that your `OPENAI_API_KEY` is set and try again."
            )
            st.error(str(exc))
            return None

    if not index_path.exists():
        st.error("Vector index still missing after rebuild attempt.")
        return None

    return index_path.stat().st_mtime

def initialize_vector_db():
    global db_books
    index_version = ensure_vector_index()
    if index_version is None:
        db_books = None
        return
    db_books = load_vector_db(index_version)

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16):
    try:
        if db_books is None:
            st.error("Vector index not available. Check the setup logs above and try again.")
            return pd.DataFrame()
        # Get semantic recommendations from vector database
        recs = db_books.similarity_search(query, k=initial_top_k)
        
        # Extract deduplicated ISBNs from vector metadata
        books_list = []
        seen_isbns = set()
        for rec in recs:
            isbn_str = rec.metadata.get("isbn13")
            if not isbn_str:
                continue
            isbn_str = str(isbn_str).strip()
            if not isbn_str or isbn_str in seen_isbns:
                continue
            seen_isbns.add(isbn_str)
            books_list.append(isbn_str)
        
        if not books_list:
            return pd.DataFrame()  # Return empty DataFrame if no ISBNs found
        
        # Filter books by ISBN
        indexed_books = books.set_index("isbn13")
        book_recs = indexed_books.reindex(books_list).dropna(how="all").reset_index()
        book_recs.rename(columns={"index": "isbn13"}, inplace=True)
        
        if len(book_recs) == 0:
            return pd.DataFrame()  # Return empty if no matches
        
        # Apply category filter
        if category and category != "All":
            book_recs = book_recs[book_recs["simple_category"] == category].copy()
            if len(book_recs) == 0:
                return pd.DataFrame()
        
        # Limit to final_top_k before sorting
        book_recs = book_recs.head(final_top_k).copy()
        
        # Sort by emotional tone if specified
        if tone and tone != "All":
            if tone == "Happy" and "joy" in book_recs.columns:
                book_recs.sort_values(by="joy", ascending=False, inplace=True)
            elif tone == "Surprising" and "surprise" in book_recs.columns:
                book_recs.sort_values(by="surprise", ascending=False, inplace=True)
            elif tone == "Angry" and "anger" in book_recs.columns:
                book_recs.sort_values(by="anger", ascending=False, inplace=True)
            elif tone == "Suspenseful" and "fear" in book_recs.columns:
                book_recs.sort_values(by="fear", ascending=False, inplace=True)
            elif tone == "Sad" and "sadness" in book_recs.columns:
                book_recs.sort_values(by="sadness", ascending=False, inplace=True)
        
        return book_recs
    
    except Exception as e:
        st.error(f"Error retrieving recommendations: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.set_page_config(
    page_title="Semantic Book Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_vector_db()

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    **Search for books using natural language:**
    - Describe what you're looking for in plain English
    - Examples: "a book about personal growth", "a mystery novel set in Victorian London"
    
    **Filter by category:**
    - Choose Fiction, Nonfiction, or Children's books
    - Select "All" to search across all categories
    
    **Sort by emotional tone:**
    - Find books that match your mood
    - Options: Happy, Surprising, Angry, Suspenseful, or Sad
    - Select "All" for no emotional filtering
    """)
    
    st.divider()
    
    st.header("About")
    st.markdown("""
    This recommender uses **semantic search** powered by AI to understand the meaning behind your queries, not just keywords.
    
    The system uses:
    - **Vector embeddings** for semantic understanding
    - **Zero-shot classification** for automatic categorization
    - **Sentiment analysis** for emotion-based sorting
    """)
    
    st.divider()
    
    st.markdown(f"**Dataset:** {len(books):,} books available")
    st.markdown(f"**Categories:** {len(books['simple_category'].dropna().unique())} types")

# Main content
st.title("Semantic Book Recommender")
st.markdown("Discover books that match your interests using natural language search powered by AI")

st.divider()

# Input section
st.subheader("Search & Filter")
with st.container():
    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
    with col1:
        user_query = st.text_input(
            "Describe the book you're looking for:",
            placeholder="e.g., A story about forgiveness and redemption",
            label_visibility="visible",
            help="Use natural language to describe what you want. The AI understands context and meaning."
        )
    with col2:
        categories = ["All"] + sorted(books["simple_category"].dropna().unique().tolist())
        category = st.selectbox(
            "Category:",
            categories,
            label_visibility="visible",
            help="Filter results by book category"
        )
    with col3:
        tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
        tone = st.selectbox(
            "Emotional Tone:",
            tones,
            label_visibility="visible",
            help="Sort results by the dominant emotional tone"
        )
    with col4:
        st.write("")  # Spacer
        st.write("")  # Spacer
        find_button = st.button("Search", use_container_width=True, type="primary")

# Display recommendations
if find_button or user_query:
    if user_query:
        with st.spinner("Searching for recommendations..."):
            try:
                recommendations = retrieve_semantic_recommendations(user_query, category, tone)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                recommendations = pd.DataFrame()
        
        if len(recommendations) > 0:
            st.divider()
            st.subheader(f"Recommendations ({len(recommendations)} found)")
            st.markdown("*Results are ranked by semantic similarity to your query*")
            
            # Display in grid layout (4 columns for better mobile experience)
            num_cols = 4
            num_rows = (len(recommendations) + num_cols - 1) // num_cols
            
            for row_idx in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    book_idx = row_idx * num_cols + col_idx
                    if book_idx < len(recommendations):
                        row = recommendations.iloc[book_idx]
                        with cols[col_idx]:
                            # Book card with border
                            with st.container():
                                try:
                                    st.image(row["large_thumbnail"], width=150, use_container_width=False)
                                except:
                                    st.image("cover-not-found.jpg", width=150, use_container_width=False)
                                
                                description = str(row["description"]) if pd.notna(row["description"]) else "No description available"
                                truncated_desc = " ".join(description.split()[:30]) + "..." if len(description.split()) > 30 else description
                                
                                authors = str(row["authors"]) if pd.notna(row["authors"]) else "Unknown"
                                authors_split = authors.split(";")
                                if len(authors_split) == 2:
                                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                                elif len(authors_split) > 2:
                                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                                else:
                                    authors_str = authors
                                
                                st.markdown(f"**{row['title']}**")
                                st.markdown(f"*by {authors_str}*")
                                st.caption(truncated_desc)
                                
                                # Show category badge if available
                                if pd.notna(row.get("simple_category")):
                                    st.caption(f"Category: {row['simple_category']}")
        else:
            st.divider()
            st.warning("No recommendations found. Try a different query or adjust your filters.")
            st.info("**Tips for better results:**\n- Use descriptive phrases (e.g., 'a book about' instead of single words)\n- Try broader categories if using filters\n- Experiment with different emotional tones")
            
            # Debug info (can be removed later)
            with st.expander("Debug Information"):
                st.write(f"**Query:** {user_query}")
                st.write(f"**Category Filter:** {category}")
                st.write(f"**Tone Filter:** {tone}")
                st.write(f"**Total books in dataset:** {len(books):,}")
                st.write(f"**Vector DB collection count:** {db_books._collection.count() if hasattr(db_books, '_collection') else 'N/A'}")
    else:
        st.divider()
        st.info("Enter a book description above to get started. Use natural language to describe what you're looking for.")
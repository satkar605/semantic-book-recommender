import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st

load_dotenv()

# Cache data loading for better performance
@st.cache_data
def load_books():
    books = pd.read_csv("data/books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    db_books = Chroma(
        persist_directory="data/chroma_index",
        embedding_function=embeddings
    )
    return db_books

# Load dataset and vector database
books = load_books()
db_books = load_vector_db()

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16):
    try:
        # Get semantic recommendations from vector database
        recs = db_books.similarity_search(query, k=initial_top_k)
        
        # Extract ISBNs from search results (more robust parsing)
        books_list = []
        for rec in recs:
            try:
                # Try to extract ISBN from the first part of the page content
                isbn_str = rec.page_content.strip().strip('"').strip("'").split()[0]
                # Remove any quotes and convert to int
                isbn_str = isbn_str.strip('"').strip("'")
                if isbn_str.isdigit():
                    books_list.append(int(isbn_str))
            except (ValueError, IndexError, AttributeError) as e:
                # Skip this record if we can't parse the ISBN
                continue
        
        if not books_list:
            return pd.DataFrame()  # Return empty DataFrame if no ISBNs found
        
        # Filter books by ISBN
        book_recs = books[books["isbn13"].isin(books_list)].copy()
        
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
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
books_df = pd.read_csv("goodreads_works.csv", low_memory=False)
reviews_df = pd.read_csv("goodreads_reviews.csv", low_memory=False)

# Clean reviews
reviews_df = reviews_df[['work_id', 'rating', 'review_text']].dropna(subset=['review_text'])
reviews_df['review_text'] = reviews_df['review_text'].astype(str)

# Clean books
books_df = books_df[['work_id', 'original_title', 'author', 'avg_rating', 'ratings_count', 'description', 'image_url']]
books_df = books_df.dropna(subset=['original_title'])
books_df = books_df.drop_duplicates(subset=['original_title'])

# Merge reviews with books
merged_df = pd.merge(
    books_df,
    reviews_df.groupby('work_id')['review_text'].apply(lambda x: ' '.join(x[:3])).reset_index(),
    on='work_id',
    how='left'
)

# Recommender
def get_recommendations(title, df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['original_title'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = df[df['original_title'].str.lower() == title.lower()].index
    if idx.empty:
        return pd.DataFrame()
    
    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][['original_title', 'author', 'avg_rating', 'description', 'review_text', 'image_url']]

# Streamlit UI
st.set_page_config(page_title="📚 Maven Bookshelf", layout="wide")
st.title("📚 Maven Bookshelf – Build Your Ideal Summer Reading List")
st.subheader("By: Mohammed Mirzan")
st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("📘 Total Books", f"{books_df['original_title'].nunique():,}")
col2.metric("✍️ Total Authors", f"{books_df['author'].nunique():,}")
col3.metric("⭐ Avg Rating", f"{books_df['avg_rating'].mean():.2f}")
col4.metric("🗣 Total Reviews", f"{reviews_df.shape[0]:,}")
st.markdown("---")

st.image("dashboard_screenshot.png", caption="📊 Interactive Dashboard Preview", use_column_width=True)

# Recommendation
st.header("🔍 Search for Similar Books")
book_input = st.text_input("Enter a book title (e.g. 'Congo')", "")

if book_input:
    recs = get_recommendations(book_input, merged_df)
    if not recs.empty:
        st.subheader(f"📖 If you liked *{book_input}*, you might also enjoy:")
        for _, row in recs.iterrows():
            st.markdown(f"### {row['original_title']} by *{row['author']}*")
            if pd.notna(row.get('image_url', None)):
                st.image(row['image_url'], width=120)
            st.markdown(f"**Rating:** ⭐ {row['avg_rating']}")
            st.markdown(f"**Summary:** _{row['description'][:300] if pd.notna(row['description']) else 'No description available'}..._")
            if pd.notna(row['review_text']):
                st.markdown(f"**Readers say:** _{row['review_text'][:300]}..._")
            st.markdown("---")
    else:
        st.warning("Book not found. Try another title.")

# Top-rated books
st.header("📚 Explore Top Books")
top_books = books_df.sort_values(by='avg_rating', ascending=False).head(10)
for _, row in top_books.iterrows():
    st.markdown(f"**{row['original_title']}** by {row['author']} — ⭐ {row['avg_rating']}")

st.markdown("---")
st.caption("Built for the Maven Bookshelf Challenge – Summer 2025 Edition")

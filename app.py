import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
    layout="centered"
)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/movie_industry.csv")
    return df

movies = load_data()

# -------------------------
# STANDARDIZE COLUMNS
# -------------------------
movies.columns = movies.columns.str.lower()

if 'movie' in movies.columns:
    movies.rename(columns={'movie': 'title'}, inplace=True)
if 'name' in movies.columns:
    movies.rename(columns={'name': 'title'}, inplace=True)

# Determine description column
desc_col = None
for col in ['description', 'overview', 'plot', 'synopsis']:
    if col in movies.columns:
        desc_col = col
        break

if desc_col is None:
    movies['description'] = "Description not available"
else:
    movies['description'] = movies[desc_col].fillna("Description not available")

# Standardize other columns
for col in ['genre', 'director', 'actors', 'stars', 'rating']:
    if col not in movies.columns:
        movies[col] = ''
    movies[col] = movies[col].fillna('')

movies = movies[movies['title'] != '']

# -------------------------
# FEATURE ENGINEERING
# -------------------------
movies['tags'] = (
    movies['genre'] + " " +
    movies['description'] + " " +
    movies['director'] + " " +
    movies['actors'] + " " +
    movies['stars']
).str.lower()

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# -------------------------
# RECOMMEND FUNCTION
# -------------------------
def recommend(movie_title):
    index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]] for i in movie_list]

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: black;
    color: white;
}

/* Get Recommendations button */
.stButton>button:first-of-type {
    background-color: lightblue;
    color: white;
    border-radius: 12px;
    padding: 0.7em 1.4em;
    font-weight: bold;
    border: none;
    transition: transform 0.2s ease;
}
.stButton>button:first-of-type:hover {
    background-color: #333333;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE AND TAGLINE
# -------------------------
st.markdown("""
<div style="
    font-size: 60px; 
    font-weight: bold; 
    text-align: center; 
    background: linear-gradient(90deg, #ff9800, #ff6f00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
">🎬 Movie Recommendation App</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align: center; 
    font-size: 24px; 
    font-weight: 600; 
    background: none;
    background-image: linear-gradient(90deg, #ff4081, #3f51b5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 30px;
">Find your vibe, find yourself, find your perfect movie 🎥</div>
""", unsafe_allow_html=True)

# -------------------------
# MOVIE SELECTBOX
# -------------------------
selected_movie = st.selectbox(
    "🎥 Choose a movie you like",
    movies['title'].values
)

# -------------------------
# GET RECOMMENDATIONS BUTTON
# -------------------------
if st.button("✨ Get Recommendations"):
    st.markdown("<h2 style='text-align:center; margin-top:20px; color:white;'>🔥 Recommended Movies</h2>", unsafe_allow_html=True)
    recommended_movies = recommend(selected_movie)
    
    for movie in recommended_movies:
        # Take first 4–5 sentences from description
        sentences = re.split(r'(?<=[.!?]) +', movie['description'])
        description_short = ' '.join(sentences[:5]) if len(sentences) > 0 else "Description not available"
        
        st.markdown(f"""
        <div style="
            background: black;
            border-left: 5px solid #ff4081;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
        ">
            <h3 style='color:#ff4081; font-size:1.5em; margin-bottom:8px;'>{movie['title']}</h3>
            <p style='color:#c7c7ff; margin:5px 0;'><strong>Genre:</strong> {movie['genre']}</p>
            <p style='color:#c7c7ff; margin:5px 0;'><strong>Director:</strong> {movie['director']}</p>
            <p style='color:#c7c7ff; margin:5px 0;'><strong>Description:</strong> {description_short}</p>
            <p style='color:#c7c7ff; margin:5px 0;'><strong>Rating:</strong> {movie['rating']}</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<div style="text-align:center; color:#999; margin-top:30px; font-size:0.9em;">
Built with Python • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)

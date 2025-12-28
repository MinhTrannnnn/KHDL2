"""
Netflix-Style Streamlit App for Hybrid Movie Recommendation System
Supports both existing users and temporary (anonymous) users
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import warnings
import re
from pages_visualization import show_data_visualization_page
warnings.filterwarnings('ignore')

# ========================================
# Page Configuration
# ========================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# Custom CSS - Netflix Style
# ========================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e50914;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #f40612;
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        background-color: #333333;
        color: white;
        border: 1px solid #555555;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #e50914;
    }
    
    /* DataFrame */
    .dataframe {
        background-color: #1a1a1a;
        color: white;
    }
    
    /* Netflix-style Genre Row */
    .genre-row {
        margin: 30px 0;
    }
    
    .genre-title {
        color: #e5e5e5;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        padding-left: 10px;
    }
    
    /* Netflix-style Movie Card */
    .netflix-card {
        background-color: #1a1a1a;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        height: 100%;
        position: relative;
        display: flex;
        flex-direction: column;
    }
    
    .netflix-card:hover {
        transform: scale(1.08);
        box-shadow: 0 8px 16px rgba(0,0,0,0.6);
        z-index: 10;
    }
    
    .netflix-card img {
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 8px 8px 0 0;
        display: block;
    }
    
    .netflix-card .poster-placeholder {
        width: 100%;
        height: 280px;
        background: #333;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px 8px 0 0;
    }
    
    .netflix-card-content {
        padding: 12px;
        background-color: #1a1a1a;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    
    .netflix-card-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 8px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        line-height: 1.3;
    }
    
    .netflix-card-rating {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.4);
    }
    
    .netflix-card-info {
        color: #b3b3b3;
        font-size: 13px;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    
    .netflix-card-genres {
        color: #46d369;
        font-size: 14px;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        line-height: 1.4;
    }
    
    /* Custom Like Button */
    .like-button {
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(229, 9, 20, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        width: 100%;
    }
    
    .like-button:hover {
        background: linear-gradient(135deg, #f40612 0%, #c9080f 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.5);
    }
    
    .like-button:active {
        transform: translateY(0);
    }
    
    .liked-button {
        background: linear-gradient(135deg, #46d369 0%, #2ea856 100%);
        box-shadow: 0 2px 8px rgba(70, 211, 105, 0.3);
    }
    
    .liked-button:hover {
        background: linear-gradient(135deg, #52e077 0%, #35c263 100%);
        box-shadow: 0 4px 12px rgba(70, 211, 105, 0.5);
    }
    
    /* Old vertical card styles - keep for search results */
    .movie-card {
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    
    .movie-card:hover {
        transform: scale(1.02);
        background-color: #222222;
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .movie-info {
        color: #b3b3b3;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .movie-genres {
        color: #46d369;
        font-size: 13px;
    }
    
    .rating-badge {
        background-color: #e50914;
        color: white;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Session history */
    .session-history {
        background-color: #222222;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* User type badge */
    .user-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    
    .existing-user {
        background-color: #46d369;
        color: #000000;
    }
    
    .temp-user {
        background-color: #ffa500;
        color: #000000;
    }
    
    /* Info box */
    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #e50914;
        padding: 15px;
        margin: 15px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# Load Models and Data (Cached)
# ========================================
@st.cache_resource
def load_models():
    """Load all models and data (cached for performance)"""
    try:
        # Load original model (individual files) - PRIORITIZED
        import os
        
        # Try loading legacy individual models FIRST
        try:
            # Load SVD model
            with open('models/svd_model.pkl', 'rb') as f:
                svd = pickle.load(f)
            
            # Load TF-IDF vectorizer
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                tfidf = pickle.load(f)
            
            # Load factors
            user_factors = np.load('models/user_factors.npy')
            item_factors = np.load('models/item_factors.npy')
            
            # Load mappings
            with open('models/movie_id_to_idx.pkl', 'rb') as f:
                movie_id_to_idx = pickle.load(f)
            
            with open('models/user_id_to_idx.pkl', 'rb') as f:
                user_id_to_idx = pickle.load(f)
            
            with open('models/tfidf_movie_id_to_row.pkl', 'rb') as f:
                tfidf_movie_id_to_row = pickle.load(f)
            
            # Load dataframes
            movies_df = pd.read_pickle('models/movies_df_clean.pkl')
            train_df = pd.read_pickle('models/train_df.pkl')
            tfidf_df = pd.read_pickle('models/tfidf_df.pkl')
            
            # Load poster URLs from CSV
            try:
                movies_with_poster = pd.read_csv('data/movies_with_posters.csv')
                # Merge poster URLs into movies_df
                movies_df = movies_df.merge(
                    movies_with_poster[['movieId', 'poster_url']], 
                    on='movieId', 
                    how='left'
                )
            except Exception as poster_error:
                # If poster file not found, add empty poster_url column
                movies_df['poster_url'] = None
                st.warning(f"⚠️ Could not load movie posters: {str(poster_error)}")
            
            # Exclude specific movies with broken posters
            # 260: Star Wars: Episode IV - A New Hope (1977)
            movies_df = movies_df[movies_df['movieId'] != 260]
            
            # Create TF-IDF matrix
            tfidf_matrix = csr_matrix(tfidf_df.values)
            
            # Calculate biases
            global_mean = train_df["rating"].mean()
            user_bias = (train_df.groupby("userId")["rating"].mean() - global_mean).to_dict()
            movie_bias = (train_df.groupby("movieId")["rating"].mean() - global_mean).to_dict()
            
            
            return {
                'svd': svd,
                'tfidf': tfidf,
                'user_factors': user_factors,
                'item_factors': item_factors,
                'movie_id_to_idx': movie_id_to_idx,
                'user_id_to_idx': user_id_to_idx,
                'tfidf_movie_id_to_row': tfidf_movie_id_to_row,
                'movies_df': movies_df,
                'train_df': train_df,
                'tfidf_matrix': tfidf_matrix,
                'global_mean': global_mean,
                'user_bias': user_bias,
                'movie_bias': movie_bias
            }
        except:
            # Fallback: Load new model with description (single file)
            if os.path.exists('models/hybrid_model_with_description.pkl'):
                st.info("⚠️ Original model not found. Loading model with description...")
                with open('models/hybrid_model_with_description.pkl', 'rb') as f:
                    artifacts = pickle.load(f)
                
                # New model already contains everything in a dictionary
                return {
                    'svd': None, # Not used for inference
                    'tfidf': None, # Not used for inference
                    'user_factors': artifacts['user_factors'],
                    'item_factors': artifacts['item_factors'],
                    'movie_id_to_idx': artifacts['movie_id_to_idx'],
                    'user_id_to_idx': artifacts['user_id_to_idx'],
                    'tfidf_movie_id_to_row': artifacts['tfidf_movie_id_to_row'],
                    'movies_df': artifacts['movies_df'],
                    'train_df': artifacts['train_df'],
                    'tfidf_matrix': artifacts['tfidf_matrix'],
                    'global_mean': artifacts['global_mean'],
                    'user_bias': artifacts['user_bias'],
                    'movie_bias': artifacts['movie_bias']
                }
            else:
                raise Exception("No model files found!")
                
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# ========================================
# Prediction Functions
# ========================================
def predict_rating_cf(user_id, movie_id, user_factors, item_factors, 
                     user_id_to_idx, movie_id_to_idx,
                     global_mean, user_bias, movie_bias):
    """Collaborative Filtering prediction"""
    bu = user_bias.get(user_id, 0.0)
    bi = movie_bias.get(movie_id, 0.0)
    
    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx:
        return global_mean + bu + bi
    
    uidx = user_id_to_idx[user_id]
    midx = movie_id_to_idx[movie_id]
    
    interaction = np.dot(user_factors[uidx], item_factors[midx])
    pred = global_mean + bu + bi + interaction
    
    return pred

def predict_rating_cb(user_id, movie_id, train_df, tfidf_matrix, 
                      tfidf_movie_id_to_row, global_mean, user_bias, movie_bias):
    """Content-Based Filtering prediction"""
    user_ratings = train_df.loc[train_df["userId"] == user_id, ["movieId", "rating"]]
    
    base = global_mean + user_bias.get(user_id, 0.0) + movie_bias.get(movie_id, 0.0)
    
    if user_ratings.empty:
        return base
    
    target_row = tfidf_movie_id_to_row.get(int(movie_id))
    if target_row is None:
        return base
    
    rated = user_ratings.copy()
    rated["row"] = rated["movieId"].map(tfidf_movie_id_to_row)
    rated = rated.dropna(subset=["row"])
    if rated.empty:
        return base
    
    rated_rows = rated["row"].astype(int).to_numpy()
    ratings = rated["rating"].to_numpy(dtype=float)
    
    sims = (tfidf_matrix[rated_rows] @ tfidf_matrix[target_row].T).toarray().ravel()
    sim_sum = np.abs(sims).sum()
    
    if sim_sum == 0:
        return base
    
    return float((sims * ratings).sum() / sim_sum)

def predict_rating_hybrid(user_id, movie_id, user_factors, item_factors, 
                         user_id_to_idx, movie_id_to_idx, train_df, 
                         tfidf_matrix, tfidf_movie_id_to_row,
                         global_mean, user_bias, movie_bias,
                         cf_weight=0.3, cb_weight=0.7):
    """Hybrid prediction combining CF and CB"""
    cf_pred = predict_rating_cf(
        user_id, movie_id, user_factors, item_factors,
        user_id_to_idx, movie_id_to_idx,
        global_mean, user_bias, movie_bias
    )
    
    cb_pred = predict_rating_cb(
        user_id, movie_id, train_df, tfidf_matrix, 
        tfidf_movie_id_to_row, global_mean, user_bias, movie_bias
    )
    
    hybrid_pred = cf_weight * cf_pred + cb_weight * cb_pred
    return hybrid_pred

def predict_rating_cb_session(movie_id, session_history, movies_df,
                              tfidf_matrix, tfidf_movie_id_to_row, 
                              global_mean, movie_bias,
                              pop_weight=0.3, cb_weight=0.7):
    """
    Content-Based prediction for temporary users based on session history
    Weighted Hybrid: 
    - 30% Popularity (Global Mean + Movie Bias)
    - 70% Content Similarity (Max Similarity to History)
    """
    # 1. Popularity Score (Base)
    base = global_mean + movie_bias.get(movie_id, 0.0)
    # Clamp base between 0.5 and 5.0
    base_score = max(0.5, min(5.0, base))
    
    if not session_history:
        return base_score
    
    target_row = tfidf_movie_id_to_row.get(int(movie_id))
    if target_row is None:
        return base_score
    
    # Get rows for session history movies
    valid_history = []
    for mid in session_history:
        row = tfidf_movie_id_to_row.get(int(mid))
        if row is not None:
            valid_history.append(row)
    
    if not valid_history:
        return base_score
    
    # Calculate similarity to history items
    rated_rows = np.array(valid_history, dtype=int)
    sims = (tfidf_matrix[rated_rows] @ tfidf_matrix[target_row].T).toarray().ravel()
    
    tfidf_max_sim = 0.0
    if len(sims) > 0:
        tfidf_max_sim = sims.max()
        
    # --- BONUS 1: Genre Jaccard Similarity ---
    genre_max_sim = 0.0
    current_genres_str = movies_df[movies_df['movieId'] == movie_id]['genres'].values[0]
    
    # --- BONUS 2: Title Similarity (Franchise Matching) ---
    title_max_sim = 0.0
    current_title = str(movies_df[movies_df['movieId'] == movie_id]['title'].values[0]).lower()
    # Remove year from title e.g. "X-Men (2000)" -> "x-men"
    current_title_clean = re.sub(r'\s*\(\d{4}\)', '', current_title).strip()
    
    if isinstance(current_genres_str, str):
        current_genres = set(current_genres_str.split('|'))
        
        for hist_id in session_history:
             # Genre Sim
             hist_genres_str = movies_df[movies_df['movieId'] == hist_id]['genres'].values[0]
             if isinstance(hist_genres_str, str):
                 hist_genres = set(hist_genres_str.split('|'))
                 intersection = len(current_genres.intersection(hist_genres))
                 union = len(current_genres.union(hist_genres))
                 if union > 0:
                     jaccard = intersection / union
                     if jaccard > genre_max_sim:
                         genre_max_sim = jaccard
            
             # Title Sim
             hist_title = str(movies_df[movies_df['movieId'] == hist_id]['title'].values[0]).lower()
             hist_title_clean = re.sub(r'\s*\(\d{4}\)', '', hist_title).strip()
             
             # Check for meaningful substring overlap (avoid short words like "The", "A")
             # If one title is contained in the other and length > 3
             if len(current_title_clean) > 3 and len(hist_title_clean) > 3:
                 if current_title_clean in hist_title_clean or hist_title_clean in current_title_clean:
                     # Strong Title Match (Franchise)
                     if title_max_sim < 1.0:
                         title_max_sim = 1.0
                 else:
                     # Partial word overlap?
                     cur_words = set(current_title_clean.split())
                     hist_words = set(hist_title_clean.split())
                     common_words = cur_words.intersection(hist_words)
                     # Filter out stop words (basic list)
                     stop_words = {'the', 'a', 'an', 'of', 'and', 'in', 'on', 'at', 'to', 'for'}
                     meaningful_common = common_words - stop_words
                     
                     if len(meaningful_common) > 0:
                         # 0.5 for sharing a meaningful word (e.g. "Men" in "X-Men"?)
                         # Better: "Harry Potter" sharing "Harry" and "Potter" -> strong
                         calc_sim = len(meaningful_common) / max(len(cur_words), len(hist_words))
                         if calc_sim > title_max_sim:
                             title_max_sim = calc_sim

    # Priority: Title > ID/Content > Genre
    # If Title Match is 1.0, we force top score.
    
    # 3. Weighted Hybrid Combination
    # We basically trust Title > Content > Genre > Popularity for "Next Movie" in session
    
    final_sim = max(tfidf_max_sim, genre_max_sim, title_max_sim)
    
    # Base score from popularity (0.5 to 5.0)
    # Sim score from 0.0 to 5.0
    
    # If we have a sequence match (Title sim=1.0), we want to almost guarantee this is top.
    # We assign a huge weight to it.
    
    predicted_rating = pop_weight * base_score + cb_weight * (final_sim * 5.0)
    
    # Boosts
    if title_max_sim == 1.0:
        predicted_rating += 2.0  # Huge bonus for title match
    elif final_sim > 0.8:
        predicted_rating += 1.0
         
    return float(predicted_rating)

# ========================================
# Recommendation Functions
# ========================================
def get_recommendations(user_id, top_k, models, session_history=None, is_temp_user=False):
    """
    Get top-K movie recommendations
    
    Args:
        user_id: User ID (int or None for temp users)
        top_k: Number of recommendations
        models: Dictionary of loaded models
        session_history: List of movie IDs viewed in session
        is_temp_user: Boolean indicating if this is a temporary user
    
    Returns:
        DataFrame with recommendations
    """
    movies_df = models['movies_df']
    train_df = models['train_df']
    
    # Get candidate movies (movies not yet rated by the user)
    if is_temp_user or user_id not in models['user_id_to_idx']:
        # Temporary user or new user: all movies are candidates
        if session_history:
            # Exclude movies already viewed in session
            candidate_movies = movies_df[~movies_df['movieId'].isin(session_history)]
        else:
            candidate_movies = movies_df.copy()
    else:
        # Existing user: exclude movies already rated
        user_rated = train_df[train_df['userId'] == user_id]['movieId'].unique()
        if session_history:
            # Also exclude session history
            exclude_movies = set(user_rated) | set(session_history)
            candidate_movies = movies_df[~movies_df['movieId'].isin(exclude_movies)]
        else:
            candidate_movies = movies_df[~movies_df['movieId'].isin(user_rated)]
    
    if len(candidate_movies) == 0:
        return pd.DataFrame()
    
    # Predict ratings for all candidates
    predictions = []
    
    for _, movie in candidate_movies.iterrows():
        movie_id = movie['movieId']
        
        if is_temp_user:
            # Temporary user: use session-based content filtering or popularity
            if session_history:
                pred = predict_rating_cb_session(
                    movie_id, session_history, movies_df,
                    models['tfidf_matrix'], models['tfidf_movie_id_to_row'],
                    models['global_mean'], models['movie_bias']
                )
            else:
                # No session history: use popularity (avg rating + movie bias)
                pred = models['global_mean'] + models['movie_bias'].get(movie_id, 0.0)
        else:
            # Existing user: use hybrid prediction
            pred = predict_rating_hybrid(
                user_id, movie_id,
                models['user_factors'], models['item_factors'],
                models['user_id_to_idx'], models['movie_id_to_idx'],
                train_df, models['tfidf_matrix'], models['tfidf_movie_id_to_row'],
                models['global_mean'], models['user_bias'], models['movie_bias']
            )
        

        
        # Don't clamp yet! Allow scores > 5.0 to influence sorting order
        # pred = max(0.5, min(5.0, pred))
        
        pred_dict = {
            'movieId': movie_id,
            'title': movie['title'],
            'genres': movie['genres'],
            'predicted_rating': pred,
            'avg_rating': movie.get('avg_rating', 0),
            'rating_count': movie.get('rating_count', 0)
        }
        
        # Add poster_url if available
        if 'poster_url' in movie.index:
            pred_dict['poster_url'] = movie['poster_url']
        else:
            pred_dict['poster_url'] = None
            
        predictions.append(pred_dict)
    
    # Sort by predicted rating and get top-K
    recommendations = pd.DataFrame(predictions)
    recommendations = recommendations.sort_values('predicted_rating', ascending=False).head(top_k)
    recommendations = recommendations.reset_index(drop=True)
    
    # Clamp ratings for display purposes ONLY after sorting
    recommendations['predicted_rating'] = recommendations['predicted_rating'].clip(0.5, 5.0)
    
    # Re-rank based on session history if available
    if session_history and len(session_history) > 0:
        recommendations = rerank_by_session_similarity(
            recommendations, session_history, models
        )
    
    return recommendations

def rerank_by_session_similarity(recommendations, session_history, models):
    """
    Re-rank recommendations based on similarity to recently viewed items
    """
    if len(session_history) == 0:
        return recommendations
    
    tfidf_matrix = models['tfidf_matrix']
    tfidf_movie_id_to_row = models['tfidf_movie_id_to_row']
    
    # Calculate average similarity to session history
    similarities = []
    
    for _, movie in recommendations.iterrows():
        movie_id = movie['movieId']
        target_row = tfidf_movie_id_to_row.get(int(movie_id))
        
        if target_row is None:
            similarities.append(0.0)
            continue
        
        # Calculate similarity to each movie in session history
        sims_to_history = []
        for hist_movie_id in session_history[-5:]:  # Consider last 5 viewed movies
            hist_row = tfidf_movie_id_to_row.get(int(hist_movie_id))
            if hist_row is not None:
                sim = (tfidf_matrix[hist_row] @ tfidf_matrix[target_row].T).toarray()[0][0]
                sims_to_history.append(sim)
        
        avg_sim = np.mean(sims_to_history) if sims_to_history else 0.0
        similarities.append(avg_sim)
    
    recommendations['session_similarity'] = similarities
    
    # Combine predicted rating and session similarity (weighted)
    recommendations['final_score'] = (
        0.7 * recommendations['predicted_rating'] + 
        0.3 * recommendations['session_similarity'] * 5.0  # Scale similarity to rating range
    )
    
    # Re-sort by final score
    recommendations = recommendations.sort_values('final_score', ascending=False)
    recommendations = recommendations.reset_index(drop=True)
    
    return recommendations

def get_movie_details(movie_id, models):
    """Get movie details by ID"""
    movies_df = models['movies_df']
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if len(movie) == 0:
        return None
    
    return movie.iloc[0]

# ========================================
# Initialize Session State
# ========================================
if 'session_history' not in st.session_state:
    st.session_state.session_history = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None

if 'is_temp_user' not in st.session_state:
    st.session_state.is_temp_user = True

# ========================================
# Load Models
# ========================================
if not st.session_state.models_loaded:
    with st.spinner('🎬 Loading recommendation models...'):
        models = load_models()
        if models is not None:
            st.session_state.models = models
            st.session_state.models_loaded = True
        else:
            st.error("❌ Failed to load models. Please check that all model files exist in the 'models' folder.")
            st.stop()

models = st.session_state.models

# ========================================
# Sidebar Navigation
# ========================================
st.sidebar.title("🎬 Movie Recommender")

page = st.sidebar.radio(
    "Navigate", 
    ["Home", "Liked Movies", "Data Visualization"], 
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# User ID Input
if page == "Home":
    st.sidebar.subheader("User Settings")
    
    # Get list of existing user IDs
    existing_users = sorted(models['train_df']['userId'].unique().tolist())
    
    # User selection mode
    user_mode = st.sidebar.radio(
        "User Mode",
        ["Anonymous", "Select Existing User", "Enter User ID"],
        index=0
    )
    
    if user_mode == "Anonymous":
        user_id = None
        is_temp_user = True
        st.session_state.is_temp_user = True
        st.sidebar.markdown('<div class="user-badge temp-user">👤 Anonymous User</div>', unsafe_allow_html=True)
    
    elif user_mode == "Select Existing User":
        selected_user = st.sidebar.selectbox(
            "Choose a user:",
            options=existing_users,
            index=0,
            help="Select from existing users in the dataset"
        )
        user_id = int(selected_user)
        is_temp_user = False
        st.session_state.is_temp_user = False
        st.sidebar.markdown(f'<div class="user-badge existing-user">✓ User {user_id}</div>', unsafe_allow_html=True)
    
    else:  # Enter User ID
        user_id_input = st.sidebar.text_input(
            "Enter User ID:",
            value="",
            help="Enter a user ID number"
        )
        
        if user_id_input.strip() == "":
            user_id = None
            is_temp_user = True
            st.session_state.is_temp_user = True
            st.sidebar.warning("⚠️ No ID entered. Using anonymous mode.")
        else:
            try:
                user_id = int(user_id_input)
                is_temp_user = False
                st.session_state.is_temp_user = False
                
                # Check if user exists in dataset
                if user_id in models['user_id_to_idx']:
                    st.sidebar.markdown(f'<div class="user-badge existing-user">✓ User {user_id}</div>', unsafe_allow_html=True)
                else:
                    st.sidebar.warning("⚠️ User ID not found. Treating as new user.")
                    is_temp_user = True
                    st.session_state.is_temp_user = True
            except ValueError:
                st.sidebar.error("❌ Invalid User ID. Using anonymous mode.")
                user_id = None
                is_temp_user = True
                st.session_state.is_temp_user = True

    st.session_state.current_user_id = user_id

    # Top-K Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recommendation Settings")
    top_k = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

    # Get Recommendations Button
    st.sidebar.markdown("---")
    if st.sidebar.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner('🎬 Generating recommendations...'):
            recommendations = get_recommendations(
                user_id, top_k, models,
                session_history=st.session_state.session_history,
                is_temp_user=is_temp_user
            )
            st.session_state.recommendations = recommendations

else:
    # Settings for other pages if needed
    user_id = st.session_state.current_user_id
    is_temp_user = st.session_state.is_temp_user



# ========================================
# Main Page Content
# ========================================

if page == "Home":
    st.title("🎬 Movie Recommender")

    # User info banner
    if is_temp_user:
        st.markdown("""
        <div class="info-box">
            <strong>👤 Anonymous Mode</strong><br>
            You are browsing as a temporary user. Movies you <strong>Like</strong> will be saved to your
            <strong>Liked Movies</strong> list and used to provide personalized recommendations.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <strong>✓ Logged in as User {user_id}</strong><br>
            Recommendations are based on your ratings history and preferences using our hybrid algorithm.
        </div>
        """, unsafe_allow_html=True)

    # Display recommendations
    if st.session_state.recommendations is not None and len(st.session_state.recommendations) > 0:
        st.header(f"🎯 Top {len(st.session_state.recommendations)} Recommendations for You")
        
    # ========================================
    # Search Section
    # ========================================
    st.markdown("### 🔍 Find Movies You Love")
    col_search_1, col_search_2 = st.columns([3, 1])
    with col_search_1:
        search_query = st.selectbox(
            "Search for a movie to add to your interests:",
            options=[""] + list(models['movies_df']['title'].unique()),
            index=0,
            label_visibility="collapsed",
            placeholder="Type to search movie..."
        )

    if search_query:
        # Find the movie details
        searched_movie = models['movies_df'][models['movies_df']['title'] == search_query].iloc[0]
        
        # Display searched movie card
        s_col1, s_col2, s_col3 = st.columns([1, 4, 1])
        
        with s_col1:
            if 'poster_url' in searched_movie.index and pd.notna(searched_movie['poster_url']):
                try:
                    st.image(searched_movie['poster_url'], width=150)
                except:
                    st.markdown("🎬")
            else:
                st.markdown("🎬\n\n*No poster*")
                
        with s_col2:
            st.markdown(f"""
            <div class="movie-card" style="border: 2px solid #46d369;">
                <div class="movie-title">{searched_movie['title']}</div>
                <div class="movie-info">Movie ID: {int(searched_movie['movieId'])}</div>
                <div class="movie-genres">🎭 {searched_movie['genres']}</div>
                <div class="movie-info">
                    ⭐ Avg Rating: {searched_movie['avg_rating']:.2f} / 5.0
                </div>
                <div style="color: #46d369; font-size: 14px; margin-top: 5px;">
                    <em>Select 'Like' to help us recommend better movies!</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with s_col3:
            # Add spacing to align with card content
            st.markdown("<br>" * 3, unsafe_allow_html=True)
            
            # Check if already in history
            is_in_history = int(searched_movie['movieId']) in st.session_state.session_history
            
            if is_in_history:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #46d369 0%, #2ea856 100%); '
                    'color: white; padding: 10px; border-radius: 6px; text-align: center; '
                    'font-weight: bold; box-shadow: 0 2px 8px rgba(70, 211, 105, 0.3);'
                    '">✓ Liked</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button(f"👍 Like", key=f"like_search_{searched_movie['movieId']}", use_container_width=True, type="primary"):
                    movie_id = int(searched_movie['movieId'])
                    st.session_state.session_history.append(movie_id)
                    st.rerun()
        
        # Show similar movies using hybrid recommendation model
        st.markdown("---")
        st.markdown('<div class="genre-title">🔍 Similar Movies</div>', unsafe_allow_html=True)
        
        # Use the searched movie as "session history" to get recommendations
        searched_movie_id = int(searched_movie['movieId'])
        temp_session_history = [searched_movie_id]
        
        # Get all candidate movies (exclude the searched movie)
        candidate_movies = models['movies_df'][models['movies_df']['movieId'] != searched_movie_id].copy()
        
        # Predict ratings for all candidates using hybrid model
        predictions = []
        for _, movie in candidate_movies.iterrows():
            movie_id = int(movie['movieId'])
            
            # Use predict_rating_cb_session (content-based with session)
            predicted_rating = predict_rating_cb_session(
                movie_id=movie_id,
                session_history=temp_session_history,
                tfidf_matrix=models['tfidf_matrix'],
                tfidf_movie_id_to_row=models['tfidf_movie_id_to_row'],
                global_mean=models['global_mean'],
                movie_bias=models['movie_bias'],
                movies_df=models['movies_df']
            )
            
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': predicted_rating
            })
        
        # Convert to DataFrame and sort by predicted rating
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_rating', ascending=False).head(10)
        
        # Merge with movie details
        similar_movies = predictions_df.merge(
            models['movies_df'],
            on='movieId',
            how='left'
        )
        
        if len(similar_movies) > 0:
            # Display in Netflix-style horizontal rows
            num_movies = len(similar_movies)
            cols_per_row = 5
            
            for row_start in range(0, num_movies, cols_per_row):
                row_movies = similar_movies.iloc[row_start:row_start + cols_per_row]
                cols = st.columns(min(cols_per_row, len(row_movies)))
                
                for col, (_, movie) in zip(cols, row_movies.iterrows()):
                    with col:
                        # Movie card with poster
                        poster_html = ""
                        if 'poster_url' in movie.index and pd.notna(movie['poster_url']):
                            poster_html = f'<img src="{movie["poster_url"]}" alt="{movie["title"]}">'
                        else:
                            poster_html = '<div class="poster-placeholder"><span style="font-size: 48px;">🎬</span></div>'
                        
                        # Truncate title if too long
                        display_title = movie['title']
                        if len(display_title) > 30:
                            display_title = display_title[:27] + '...'
                        
                        card_html = f"""
                        <div class="netflix-card">
                            {poster_html}
                            <div class="netflix-card-content">
                                <div class="netflix-card-title" title="{movie['title']}">{display_title}</div>
                                <div class="netflix-card-info">⭐ {movie['avg_rating']:.1f}/5.0</div>
                                <div class="netflix-card-genres" title="{movie['genres']}">{movie['genres']}</div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Small Like button
                        movie_id = int(movie['movieId'])
                        is_liked = movie_id in st.session_state.session_history
                        
                        if is_liked:
                            st.markdown(
                                '<div style="background: linear-gradient(135deg, #46d369 0%, #2ea856 100%); '
                                'color: white; padding: 6px 12px; border-radius: 6px; text-align: center; '
                                'font-weight: bold; font-size: 12px; box-shadow: 0 2px 8px rgba(70, 211, 105, 0.3);'
                                '">✓ Liked</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            if st.button("👍", key=f"like_similar_{movie_id}", use_container_width=True, type="primary", help="Like this movie"):
                                if movie_id not in st.session_state.session_history:
                                    st.session_state.session_history.append(movie_id)
                                st.toast(f"✓ Liked {movie['title']}!", icon="🎬")
                                st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No similar movies found.")

    st.markdown("---")

    # Display recommendations
    if st.session_state.recommendations is not None and len(st.session_state.recommendations) > 0:
        recommendations = st.session_state.recommendations
        
        # Display all movies in Netflix-style rows (no genre grouping)
        st.markdown('<div class="genre-title">🎯 Recommended for You</div>', unsafe_allow_html=True)
        
        num_movies = len(recommendations)
        cols_per_row = 5
        
        # Split into rows of 5 movies each
        for row_start in range(0, num_movies, cols_per_row):
            row_movies = recommendations.iloc[row_start:row_start + cols_per_row]
            cols = st.columns(min(cols_per_row, len(row_movies)))
            
            for col, (_, movie) in zip(cols, row_movies.iterrows()):
                with col:
                    # Movie card with poster
                    poster_html = ""
                    if 'poster_url' in movie.index and pd.notna(movie['poster_url']):
                        poster_html = f'<img src="{movie["poster_url"]}" alt="{movie["title"]}">'
                    else:
                        poster_html = '<div class="poster-placeholder"><span style="font-size: 48px;">🎬</span></div>'
                    
                    # Truncate title if too long
                    display_title = movie['title']
                    if len(display_title) > 30:
                        display_title = display_title[:27] + '...'
                    
                    card_html = f"""
                    <div class="netflix-card">
                        {poster_html}
                        <div class="netflix-card-content">
                            <div class="netflix-card-title" title="{movie['title']}">{display_title}</div>
                            <div class="netflix-card-info">⭐ {movie['avg_rating']:.1f}/5.0 ({int(movie['rating_count'])} ratings)</div>
                            <div class="netflix-card-genres" title="{movie['genres']}">{movie['genres']}</div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Custom Like button with better styling
                    movie_id = int(movie['movieId'])
                    is_liked = movie_id in st.session_state.session_history
                    
                    button_html = f"""
                    <style>
                        div[data-testid="stButton"] button[kind="primary"] {{
                            background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
                            border: none;
                            border-radius: 6px;
                            padding: 6px 12px;
                            font-weight: bold;
                            font-size: 12px;
                            transition: all 0.3s ease;
                            box-shadow: 0 2px 8px rgba(229, 9, 20, 0.3);
                        }}
                        div[data-testid="stButton"] button[kind="primary"]:hover {{
                            background: linear-gradient(135deg, #f40612 0%, #c9080f 100%);
                            transform: translateY(-2px);
                            box-shadow: 0 4px 12px rgba(229, 9, 20, 0.5);
                        }}
                    </style>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)
                    
                    if is_liked:
                        st.markdown(
                            '<div style="background: linear-gradient(135deg, #46d369 0%, #2ea856 100%); '
                            'color: white; padding: 6px 12px; border-radius: 6px; text-align: center; '
                            'font-weight: bold; font-size: 12px; box-shadow: 0 2px 8px rgba(70, 211, 105, 0.3);'
                            '">✓ Liked</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        if st.button("👍 Like", key=f"like_{movie_id}", use_container_width=True, type="primary"):
                            if movie_id not in st.session_state.session_history:
                                st.session_state.session_history.append(movie_id)
                            if is_temp_user:
                                st.toast("✓ Liked! Get new recommendations to see similar movies!", icon="🎬")
                            st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Summary statistics at the bottom
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Predicted Rating", 
                f"{recommendations['predicted_rating'].mean():.2f}"
            )
        
        with col2:
            st.metric(
                "Total Recommendations",
                len(recommendations)
            )
        
        with col3:
            if is_temp_user:
                st.metric(
                    "Movies Liked",
                    len(st.session_state.session_history)
                )
            else:
                # Count user's ratings in train set
                user_ratings = len(models['train_df'][models['train_df']['userId'] == user_id])
                st.metric(
                    "Your Total Ratings",
                    user_ratings
                )

    else:
        # No recommendations yet
        pass
        
        # Show some popular movies as examples
        st.markdown('<div class="genre-title">🔥 Popular Movies</div>', unsafe_allow_html=True)
        
        # Select columns that exist
        cols_to_select = ['movieId', 'title', 'genres', 'avg_rating', 'rating_count']
        if 'poster_url' in models['movies_df'].columns:
            cols_to_select.append('poster_url')
        
        popular_movies = models['movies_df'].nlargest(20, 'rating_count')[cols_to_select]
        
        # Display in Netflix-style horizontal rows (5 per row)
        num_movies = len(popular_movies)
        cols_per_row = 5
        
        for row_start in range(0, num_movies, cols_per_row):
            row_movies = popular_movies.iloc[row_start:row_start + cols_per_row]
            cols = st.columns(min(cols_per_row, len(row_movies)))
            
            for col, (_, movie) in zip(cols, row_movies.iterrows()):
                with col:
                    # Movie card with poster
                    poster_html = ""
                    if 'poster_url' in movie.index and pd.notna(movie['poster_url']):
                        poster_html = f'<img src="{movie["poster_url"]}" alt="{movie["title"]}">'
                    else:
                        poster_html = '<div class="poster-placeholder"><span style="font-size: 48px;">🎬</span></div>'
                    
                    # Truncate title if too long
                    display_title = movie['title']
                    if len(display_title) > 30:
                        display_title = display_title[:27] + '...'
                    
                    card_html = f"""
                    <div class="netflix-card">
                        {poster_html}
                        <div class="netflix-card-content">
                            <div class="netflix-card-title" title="{movie['title']}">{display_title}</div>
                            <div class="netflix-card-info">📊 {int(movie['rating_count'])} ratings</div>
                            <div class="netflix-card-genres" title="{movie['genres']}">{movie['genres']}</div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Like button below the card
                    movie_id = int(movie['movieId'])
                    is_liked = movie_id in st.session_state.session_history
                    
                    if is_liked:
                        st.success("✓ Liked", icon="✅")
                    else:
                        if st.button(f"👍 Like", key=f"like_popular_{movie_id}", use_container_width=True):
                            if movie_id not in st.session_state.session_history:
                                st.session_state.session_history.append(movie_id)
                            st.success(f"✓ Liked!")
                            if is_temp_user:
                                st.info("💡 Get recommendations to see movies similar to this!")
                            st.rerun()

elif page == "Liked Movies":
    st.title("❤️ Liked Movies")
    
    if not st.session_state.session_history:
        st.info("You haven't liked any movies yet. Go to Home to explore and like movies!")
    else:
        st.write(f"You have liked {len(st.session_state.session_history)} movies.")
        if st.button("🗑️ Clear All Liked Movies"):
            st.session_state.session_history = []
            st.rerun()
            
        st.markdown("---")
        
        # Display liked movies in a grid or list
        for movie_id in st.session_state.session_history[::-1]:
            movie = get_movie_details(movie_id, models)
            
            if movie is not None:
                col1, col2 = st.columns([1, 5])
                
                with col1:
                    if 'poster_url' in movie.index and pd.notna(movie['poster_url']):
                        try:
                            st.image(movie['poster_url'], width=100)
                        except:
                            st.markdown("🎬")
                    else:
                        st.markdown("🎬")
                
                with col2:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-genres">🎭 {movie['genres']}</div>
                        <div class="movie-info">
                            ⭐ Avg Rating: {movie.get('avg_rating', 0):.2f} / 5.0
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


elif page == "Data Visualization":
    show_data_visualization_page()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666;">
    <p>🎬 Powered by Hybrid Recommendation System (CF + CB)</p>
    <p style="font-size: 12px;">Built with Streamlit • Data from MovieLens</p>
</div>
""", unsafe_allow_html=True)


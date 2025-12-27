"""
Streamlit App cho Hệ Gợi ý Hybrid
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Thiết lập trang
st.set_page_config(
    page_title="Hệ Gợi ý Phim Hybrid",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .rating-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background-color: #ff6b6b;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load tất cả các models đã train"""
    models = {}
    
    try:
        # Load SVD model
        with open('models/svd_model.pkl', 'rb') as f:
            models['svd'] = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            models['tfidf'] = pickle.load(f)
        
        # Load factors
        models['user_factors'] = np.load('models/user_factors.npy')
        models['item_factors'] = np.load('models/item_factors.npy')
        # content_similarity_matrix là optional (bản tối ưu không còn cần file này)
        if os.path.exists('models/content_similarity_matrix.npy'):
            models['content_similarity_matrix'] = np.load('models/content_similarity_matrix.npy')
        else:
            models['content_similarity_matrix'] = None
        
        # Load mappings
        with open('models/movie_id_to_idx.pkl', 'rb') as f:
            models['movie_id_to_idx'] = pickle.load(f)
        
        with open('models/user_id_to_idx.pkl', 'rb') as f:
            models['user_id_to_idx'] = pickle.load(f)
        
        with open('models/train_movie_ids.pkl', 'rb') as f:
            models['train_movie_ids'] = pickle.load(f)
        # Precompute mapping để tránh list.index() O(n)
        models['train_movie_id_to_idx'] = {int(mid): i for i, mid in enumerate(models['train_movie_ids'])}
        
        # Load dataframes
        models['movies_df'] = pd.read_pickle('models/movies_df_clean.pkl')
        models['train_df'] = pd.read_pickle('models/train_df.pkl')
        models['tfidf_df'] = pd.read_pickle('models/tfidf_df.pkl')

        # Load TF-IDF movieId -> row mapping nếu có, fallback: tạo từ tfidf_df.index
        if os.path.exists('models/tfidf_movie_id_to_row.pkl'):
            with open('models/tfidf_movie_id_to_row.pkl', 'rb') as f:
                models['tfidf_movie_id_to_row'] = pickle.load(f)
        else:
            models['tfidf_movie_id_to_row'] = {int(mid): i for i, mid in enumerate(models['tfidf_df'].index.values)}
        
        return models, None
    except FileNotFoundError as e:
        return None, f"Không tìm thấy file model: {e}\n\nVui lòng chạy script train_models.py trước!"

def predict_rating_cf(user_id, movie_id, models):
    """Dự đoán rating sử dụng Collaborative Filtering"""
    user_factors = models['user_factors']
    item_factors = models['item_factors']
    user_id_to_idx = models['user_id_to_idx']
    movie_id_to_idx = models['movie_id_to_idx']
    train_df = models['train_df']
    
    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx:
        return train_df['rating'].mean()
    
    user_idx = user_id_to_idx[user_id]
    movie_idx = movie_id_to_idx[movie_id]
    
    prediction = np.dot(user_factors[user_idx], item_factors[movie_idx])
    prediction = np.clip(prediction, 0.5, 5.0)
    return prediction

def predict_rating_cb(user_id, movie_id, models):
    """Dự đoán rating sử dụng Content-Based Filtering"""
    train_df = models['train_df']
    content_similarity_matrix = models.get('content_similarity_matrix')
    train_movie_id_to_idx = models.get('train_movie_id_to_idx', {})
    tfidf_df = models['tfidf_df']
    tfidf_movie_id_to_row = models.get('tfidf_movie_id_to_row', {})

    user_ratings = train_df.loc[train_df['userId'] == user_id, ['movieId', 'rating']]
    if user_ratings.empty:
        return float(train_df['rating'].mean())

    # Case 1: Có precomputed content_similarity_matrix (legacy)
    if content_similarity_matrix is not None:
        target_idx = train_movie_id_to_idx.get(int(movie_id))
        if target_idx is None:
            return float(train_df['rating'].mean())

        rated = user_ratings.copy()
        rated['idx'] = rated['movieId'].map(train_movie_id_to_idx)
        rated = rated.dropna(subset=['idx'])
        if rated.empty:
            return float(train_df['rating'].mean())

        rated_idxs = rated['idx'].astype(int).to_numpy()
        ratings = rated['rating'].to_numpy(dtype=float)
        sims = content_similarity_matrix[target_idx, rated_idxs]

        similarity_sum = np.abs(sims).sum()
        if similarity_sum == 0:
            return float(train_df['rating'].mean())

        pred = float((sims * ratings).sum() / similarity_sum)
        return float(np.clip(pred, 0.5, 5.0))

    # Case 2: Optimized on-the-fly similarity từ TF-IDF
    target_row = tfidf_movie_id_to_row.get(int(movie_id))
    if target_row is None:
        return float(train_df['rating'].mean())

    rated = user_ratings.copy()
    rated['row'] = rated['movieId'].map(tfidf_movie_id_to_row)
    rated = rated.dropna(subset=['row'])
    if rated.empty:
        return float(train_df['rating'].mean())

    rated_rows = rated['row'].astype(int).to_numpy()
    ratings = rated['rating'].to_numpy(dtype=float)

    # TF-IDF mặc định đã L2 normalize => cosine == dot
    tfidf_mat = tfidf_df.to_numpy(dtype=float)
    target_vec = tfidf_mat[target_row]
    sims = tfidf_mat[rated_rows] @ target_vec

    similarity_sum = np.abs(sims).sum()
    if similarity_sum == 0:
        return float(train_df['rating'].mean())

    pred = float((sims * ratings).sum() / similarity_sum)
    return float(np.clip(pred, 0.5, 5.0))

def predict_rating_hybrid(user_id, movie_id, models, cf_weight=0.6, cb_weight=0.4):
    """Dự đoán rating sử dụng Hybrid approach"""
    cf_pred = predict_rating_cf(user_id, movie_id, models)
    cb_pred = predict_rating_cb(user_id, movie_id, models)
    
    hybrid_pred = cf_weight * cf_pred + cb_weight * cb_pred
    return hybrid_pred, cf_pred, cb_pred

def get_top_recommendations(user_id, models, top_n=10, cf_weight=0.6, cb_weight=0.4):
    """Lấy top N phim được gợi ý cho user"""
    train_df = models['train_df']
    train_movie_ids = models['train_movie_ids']
    movies_df = models['movies_df']
    
    # Lấy các phim user đã đánh giá
    user_rated_movies = set(train_df[train_df['userId'] == user_id]['movieId'].tolist())
    
    # Tính prediction cho tất cả phim chưa được đánh giá
    recommendations = []
    
    for movie_id in train_movie_ids:
        if movie_id not in user_rated_movies:
            hybrid_pred, cf_pred, cb_pred = predict_rating_hybrid(
                user_id, movie_id, models, cf_weight, cb_weight
            )
            recommendations.append({
                'movieId': movie_id,
                'predicted_rating': hybrid_pred,
                'cf_rating': cf_pred,
                'cb_rating': cb_pred
            })
    
    # Sắp xếp theo predicted rating
    recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    
    # Lấy top N và merge với thông tin phim
    top_recommendations = recommendations[:top_n]
    
    result = []
    for rec in top_recommendations:
        movie_info = movies_df[movies_df['movieId'] == rec['movieId']].iloc[0]
        result.append({
            'movieId': rec['movieId'],
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'year': movie_info.get('year', 'N/A'),
            'predicted_rating': rec['predicted_rating'],
            'cf_rating': rec['cf_rating'],
            'cb_rating': rec['cb_rating']
        })
    
    return result

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Hệ Gợi ý Phim Hybrid</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    models, error = load_models()
    
    if error:
        st.error(error)
        st.info("💡 Chạy lệnh sau trong terminal để train model:\n```bash\npython train_models.py\n```")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Cài đặt")
    
    # Chọn user ID
    available_users = sorted(models['user_id_to_idx'].keys())
    selected_user = st.sidebar.selectbox(
        "Chọn User ID:",
        options=available_users,
        index=0
    )
    
    # Số lượng gợi ý
    top_n = st.sidebar.slider(
        "Số lượng phim gợi ý:",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    # Trọng số
    st.sidebar.markdown("### Trọng số Hybrid")
    cf_weight = st.sidebar.slider(
        "Collaborative Filtering:",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    cb_weight = 1.0 - cf_weight
    st.sidebar.write(f"Content-Based: {cb_weight:.1f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Gợi ý Phim", "🔍 Tìm kiếm Phim", "📊 Thống kê"])
    
    # Tab 1: Gợi ý phim
    with tab1:
        st.header(f"Top {top_n} Phim Được Gợi ý cho User {selected_user}")
        
        if st.button("🔄 Tạo Gợi ý", type="primary"):
            with st.spinner("Đang tính toán gợi ý..."):
                recommendations = get_top_recommendations(
                    selected_user, models, top_n, cf_weight, cb_weight
                )
            
            if recommendations:
                # Hiển thị kết quả
                cols = st.columns(2)
                
                for idx, rec in enumerate(recommendations):
                    col = cols[idx % 2]
                    
                    with col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>{rec['title']}</h3>
                            <p><strong>Năm:</strong> {rec['year']}</p>
                            <p><strong>Thể loại:</strong> {rec['genres']}</p>
                            <p>
                                <span class="rating-badge">⭐ {rec['predicted_rating']:.2f}</span>
                                <small> (CF: {rec['cf_rating']:.2f}, CB: {rec['cb_rating']:.2f})</small>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy phim gợi ý.")
    
    # Tab 2: Tìm kiếm và dự đoán rating
    with tab2:
        st.header("Tìm kiếm Phim và Dự đoán Rating")
        
        # Tìm kiếm phim
        movies_df = models['movies_df']
        movie_search = st.text_input("🔍 Tìm kiếm phim (theo tên):")
        
        if movie_search:
            search_results = movies_df[
                movies_df['title'].str.contains(movie_search, case=False, na=False)
            ].head(20)
            
            if len(search_results) > 0:
                selected_movie_id = st.selectbox(
                    "Chọn phim:",
                    options=search_results['movieId'].tolist(),
                    format_func=lambda x: search_results[search_results['movieId'] == x]['title'].iloc[0]
                )
                
                if selected_movie_id:
                    movie_info = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
                    
                    st.markdown("### Thông tin Phim")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Tên phim:** {movie_info['title']}")
                        st.write(f"**Năm:** {movie_info.get('year', 'N/A')}")
                    with col2:
                        st.write(f"**Thể loại:** {movie_info['genres']}")
                        if 'avg_rating' in movie_info:
                            st.write(f"**Rating trung bình:** {movie_info['avg_rating']:.2f}")
                    
                    # Dự đoán rating
                    st.markdown("### Dự đoán Rating")
                    if st.button("🔮 Dự đoán Rating", type="primary"):
                        hybrid_pred, cf_pred, cb_pred = predict_rating_hybrid(
                            selected_user, selected_movie_id, models, cf_weight, cb_weight
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Collaborative Filtering", f"{cf_pred:.2f}")
                        with col2:
                            st.metric("Content-Based", f"{cb_pred:.2f}")
                        with col3:
                            st.metric("Hybrid (Dự đoán)", f"{hybrid_pred:.2f}", 
                                     delta=f"{hybrid_pred - 3.5:.2f}")
            else:
                st.warning("Không tìm thấy phim nào.")
    
    # Tab 3: Thống kê
    with tab3:
        st.header("Thống kê")
        
        movies_df = models['movies_df']
        train_df = models['train_df']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số phim", f"{len(movies_df):,}")
        with col2:
            st.metric("Tổng số ratings", f"{len(train_df):,}")
        with col3:
            st.metric("Số lượng users", f"{train_df['userId'].nunique():,}")
        with col4:
            st.metric("Rating trung bình", f"{train_df['rating'].mean():.2f}")
        
        # Phim đã đánh giá của user
        user_ratings = train_df[train_df['userId'] == selected_user]
        
        if len(user_ratings) > 0:
            st.markdown(f"### Phim đã đánh giá của User {selected_user}")
            
            user_movies = user_ratings.merge(
                movies_df[['movieId', 'title', 'genres', 'year']],
                on='movieId',
                how='left'
            )
            
            st.dataframe(
                user_movies[['title', 'genres', 'year', 'rating']].sort_values('rating', ascending=False),
                use_container_width=True
            )
        else:
            st.info(f"User {selected_user} chưa đánh giá phim nào.")

if __name__ == "__main__":
    main()


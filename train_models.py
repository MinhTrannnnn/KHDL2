"""
Script để train và lưu các model cho hệ gợi ý hybrid
Chạy script này trước khi sử dụng Streamlit app
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Tạo thư mục models nếu chưa có
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("BẮT ĐẦU TRAIN MODEL")
print("=" * 60)

# 1. Load dữ liệu
print("\n[1/7] Đang load dữ liệu...")
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')
print(f"  - Movies: {len(movies_df)}")
print(f"  - Ratings: {len(ratings_df):,}")

# 2. Làm sạch dữ liệu
print("\n[2/7] Đang làm sạch dữ liệu...")
movies_df_clean = movies_df.dropna(subset=['title', 'genres'])
ratings_df_clean = ratings_df.dropna(subset=['userId', 'movieId', 'rating'])

# Loại bỏ duplicates
ratings_df_clean = ratings_df_clean.drop_duplicates(subset=['userId', 'movieId'], keep='last')
movies_df_clean = movies_df_clean.drop_duplicates(subset=['movieId'], keep='first')

# Xử lý outlier
ratings_df_clean = ratings_df_clean[
    (ratings_df_clean['rating'] >= 0.5) & 
    (ratings_df_clean['rating'] <= 5.0)
]

# Loại bỏ phim có ít ratings
min_ratings = 5
movie_stats = ratings_df_clean.groupby('movieId').agg({
    'rating': ['mean', 'std', 'count']
}).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'rating_std', 'rating_count']
movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
movies_df_clean = movies_df_clean.merge(movie_stats, on='movieId', how='left')
movies_df_clean = movies_df_clean[movies_df_clean['rating_count'] >= min_ratings]
ratings_df_clean = ratings_df_clean[ratings_df_clean['movieId'].isin(movies_df_clean['movieId'])]

print(f"  - Movies sau khi làm sạch: {len(movies_df_clean)}")
print(f"  - Ratings sau khi làm sạch: {len(ratings_df_clean):,}")

# 3. Tạo features
print("\n[3/7] Đang tạo features...")
import re
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return None

movies_df_clean['year'] = movies_df_clean['title'].apply(extract_year)

# One-hot encoding cho genres
all_genres = set()
for genres in movies_df_clean['genres']:
    if pd.notna(genres) and genres != '(no genres listed)':
        all_genres.update(genres.split('|'))

all_genres = sorted(list(all_genres))
for genre in all_genres:
    movies_df_clean[f'genre_{genre}'] = movies_df_clean['genres'].apply(
        lambda x: 1 if pd.notna(x) and genre in str(x) else 0
    )

print(f"  - Số genres: {len(all_genres)}")
print(f"  - Tổng số features: {len(all_genres) + 5}")

# 4. Chuẩn hóa dữ liệu
print("\n[4/7] Đang chuẩn hóa dữ liệu...")
numeric_features = ['year', 'avg_rating', 'rating_std', 'rating_count']
scaler = StandardScaler()
for feature in numeric_features:
    if feature in movies_df_clean.columns:
        movies_df_clean[f'{feature}_normalized'] = scaler.fit_transform(
            movies_df_clean[[feature]]
        )

# 5. Vector hóa (TF-IDF)
print("\n[5/7] Đang vector hóa genres (TF-IDF)...")
movies_df_clean['genres_text'] = movies_df_clean['genres'].fillna('')
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df_clean['genres_text'])

movie_indices = movies_df_clean['movieId'].values
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=movie_indices,
    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
)

print(f"  - TF-IDF Matrix shape: {tfidf_matrix.shape}")

# 6. Chia train/test và tạo user-item matrix
print("\n[6/7] Đang chia train/test và tạo user-item matrix...")
train_df, test_df = train_test_split(
    ratings_df_clean,
    test_size=0.2,
    random_state=42
)

train_matrix = train_df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

print(f"  - Train set: {len(train_df):,} ratings")
print(f"  - Test set: {len(test_df):,} ratings")
print(f"  - User-Item Matrix shape: {train_matrix.shape}")

# 7. Train Collaborative Filtering (SVD)
print("\n[7/7] Đang train Collaborative Filtering (SVD)...")
sparse_train_matrix = csr_matrix(train_matrix.values)
n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(sparse_train_matrix)
item_factors = svd.components_.T

print(f"  - SVD Components: {n_components}")
print(f"  - Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# Tạo mappings
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(train_matrix.columns)}
idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(train_matrix.index)}

# 8. Content-Based (tối ưu): không tạo full similarity matrix O(n^2)
# Lưu mapping movieId -> row index trong TF-IDF để tính similarity on-the-fly khi dự đoán.
print("\n[8/8] Đang chuẩn bị dữ liệu Content-Based (optimized)...")
train_movie_ids = train_matrix.columns.tolist()
tfidf_movie_id_to_row = {int(mid): i for i, mid in enumerate(tfidf_df.index.values)}
print("  - Đã tạo mapping TF-IDF movieId -> row index (dùng cho on-the-fly similarity).")

# Lưu tất cả các model và dữ liệu
print("\n" + "=" * 60)
print("ĐANG LƯU MODEL...")
print("=" * 60)

# Lưu models
with open('models/svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)
    
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Lưu matrices và factors
np.save('models/user_factors.npy', user_factors)
np.save('models/item_factors.npy', item_factors)

# Lưu mappings
with open('models/movie_id_to_idx.pkl', 'wb') as f:
    pickle.dump(movie_id_to_idx, f)
    
with open('models/user_id_to_idx.pkl', 'wb') as f:
    pickle.dump(user_id_to_idx, f)

with open('models/train_movie_ids.pkl', 'wb') as f:
    pickle.dump(train_movie_ids, f)

# Lưu mapping TF-IDF (để app tính similarity nhanh mà không cần content_similarity_matrix)
with open('models/tfidf_movie_id_to_row.pkl', 'wb') as f:
    pickle.dump(tfidf_movie_id_to_row, f)

# Lưu dataframes
movies_df_clean.to_pickle('models/movies_df_clean.pkl')
train_df.to_pickle('models/train_df.pkl')
tfidf_df.to_pickle('models/tfidf_df.pkl')

# Lưu scalers
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Đã lưu tất cả models thành công!")
print("\nCác file đã lưu:")
print("  - models/svd_model.pkl")
print("  - models/tfidf_vectorizer.pkl")
print("  - models/user_factors.npy")
print("  - models/item_factors.npy")
print("  - models/movie_id_to_idx.pkl")
print("  - models/user_id_to_idx.pkl")
print("  - models/train_movie_ids.pkl")
print("  - models/tfidf_movie_id_to_row.pkl")
print("  - models/movies_df_clean.pkl")
print("  - models/train_df.pkl")
print("  - models/tfidf_df.pkl")
print("  - models/scaler.pkl")
print("\n" + "=" * 60)
print("HOÀN THÀNH!")
print("=" * 60)
print("\nBây giờ bạn có thể chạy Streamlit app bằng lệnh: streamlit run app.py")


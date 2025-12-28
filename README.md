# 🎬 Netflix-Style Movie Recommendation System

A hybrid movie recommendation system with a beautiful Netflix-style Streamlit web interface.

## Features

- **Hybrid Recommendation Algorithm**: Combines Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF)
- **Dual User Support**:
  - **Existing Users**: Get personalized recommendations based on viewing history
  - **Temporary Users**: Anonymous browsing with session-based recommendations
- **Session Tracking**: Real-time tracking of viewed movies to improve recommendations
- **Smart Re-ranking**: Recommendations adapt based on recently viewed content
- **Netflix-Style UI**: Modern, dark-themed interface with movie posters

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd Film
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist** in the `models/` folder:
   - `svd_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `user_factors.npy`
   - `item_factors.npy`
   - `movie_id_to_idx.pkl`
   - `user_id_to_idx.pkl`
   - `tfidf_movie_id_to_row.pkl`
   - `movies_df_clean.pkl`
   - `train_df.pkl`
   - `tfidf_df.pkl`

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the app**: Your browser should automatically open to `http://localhost:8501`

## How to Use

### For Existing Users

1. Enter your **User ID** in the sidebar (e.g., `1`, `25`, `100`)
2. Adjust the number of recommendations using the slider (5-20)
3. Click **"Get Recommendations"**
4. Browse personalized movie recommendations
5. Click **"View"** on any movie to see more details

### For Temporary Users (Anonymous)

1. Leave the **User ID field empty**
2. Adjust the number of recommendations using the slider
3. Click **"Get Recommendations"** to get popular movies
4. Click **"View"** on movies you're interested in
5. Your viewing history is tracked in the session
6. Get recommendations again to see personalized suggestions based on what you viewed!

### Session-Based Recommendations

When browsing as a temporary user:
- Each movie you view is tracked in the current session
- The system uses content-based filtering to find similar movies
- Recommendations are re-ranked based on your viewing history
- Clear your session history anytime using the **"Clear Session History"** button

## System Architecture

### Hybrid Recommendation

The system combines two approaches:

1. **Collaborative Filtering (CF)** - 30% weight
   - Uses SVD (Singular Value Decomposition) for matrix factorization
   - Finds patterns in user-item interactions
   - Best for existing users with rating history

2. **Content-Based Filtering (CB)** - 70% weight
   - Uses TF-IDF vectorization on movie genres
   - Finds movies similar to those the user liked
   - Works for both existing and new/temporary users

### Prediction Formula

```
Hybrid_Score = 0.3 × CF_Prediction + 0.7 × CB_Prediction
```

For temporary users with session history:
```
CB_Prediction = Weighted_Average(Similarity_to_Viewed_Movies)
```

### Re-ranking

When session history exists:
```
Final_Score = 0.7 × Predicted_Rating + 0.3 × Session_Similarity
```

## Technical Details

- **Frontend**: Streamlit with custom CSS (Netflix theme)
- **Backend**: Python with NumPy, Pandas, Scikit-learn
- **Models**: Pre-trained SVD and TF-IDF models
- **Data**: MovieLens dataset with 10,000+ movies
- **Performance**: Models are cached for fast recommendations

## Dataset

- **Movies**: 10,329 movies with genres and metadata
- **Ratings**: 105,339 ratings from 668 users
- **Rating Scale**: 0.5 to 5.0 (half-star increments)

## Files Structure

```
Film/
├── streamlit_app.py         # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Dataset files
│   ├── movies.csv
│   ├── ratings.csv
│   └── movie_poster.csv
├── models/                 # Pre-trained models (required)
│   ├── svd_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_factors.npy
│   ├── item_factors.npy
│   ├── movie_id_to_idx.pkl
│   ├── user_id_to_idx.pkl
│   ├── tfidf_movie_id_to_row.pkl
│   ├── movies_df_clean.pkl
│   ├── train_df.pkl
│   └── tfidf_df.pkl
└── notebook/               # Development notebooks
    ├── test.ipynb
    ├── valuation.ipynb
    └── visualization.ipynb
```

## Troubleshooting

### Models not found
- Ensure all `.pkl` and `.npy` files are in the `models/` folder
- Check that the models were generated from the notebooks

### Streamlit not starting
- Make sure you're in the correct directory
- Check that all dependencies are installed: `pip list`
- Try: `python -m streamlit run streamlit_app.py`

### Recommendations not working
- Verify User ID exists in dataset (User IDs: 1-668)
- For temporary users, leave User ID empty
- Click "Get Recommendations" button after changing settings

## Performance Tips

- Models are cached after first load (faster subsequent recommendations)
- Session state persists during the session
- Clear session history to reset temporary user tracking

## Credits

- **Algorithm**: Hybrid Recommendation System (CF + CB)
- **Data**: MovieLens Dataset
- **UI Framework**: Streamlit
- **Style**: Inspired by Netflix

## License

This project is for educational purposes.

---

**Enjoy discovering new movies! 🍿🎬**


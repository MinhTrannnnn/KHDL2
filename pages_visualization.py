"""
Data Visualization Page for Streamlit App
Displays all charts from visualization.ipynb notebook
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_data_visualization_page():
    """Display Data Visualization page with all charts from notebook"""
    
    st.title("📊 Data Visualization")
    st.markdown("MovieLens Data Analysis and Visualization")
    
    # Load data for visualization
    try:
        # Load ratings and movies data
        ratings_df = pd.read_csv('data/ratings.csv')
        movies_df_viz = pd.read_csv('data/movies.csv')
        
        # Merge data
        df_viz = ratings_df.merge(movies_df_viz, on='movieId', how='left')
        
        # Display basic statistics
        st.markdown("### 📈 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", f"{len(movies_df_viz):,}")
        with col2:
            st.metric("Total Ratings", f"{len(ratings_df):,}")
        with col3:
            st.metric("Total Users", f"{ratings_df['userId'].nunique():,}")
        with col4:
            st.metric("Average Rating", f"{ratings_df['rating'].mean():.2f}")
        
        st.markdown("---")
        
        # 1. Rating Distribution
        st.markdown("### 📊 1. Rating Distribution")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#141414')
        
        # Bar chart
        rating_counts = df_viz['rating'].value_counts().sort_index()
        axes[0].bar(rating_counts.index, rating_counts.values, color='#e50914', alpha=0.8, edgecolor='white')
        axes[0].set_xlabel('Rating', fontsize=12, fontweight='bold', color='white')
        axes[0].set_ylabel('Number of Ratings', fontsize=12, fontweight='bold', color='white')
        axes[0].set_title('Rating Distribution - Bar Chart', fontsize=14, fontweight='bold', pad=20, color='white')
        axes[0].grid(axis='y', alpha=0.3, color='gray')
        axes[0].set_facecolor('#1a1a1a')
        axes[0].tick_params(colors='white')
        for i, v in enumerate(rating_counts.values):
            axes[0].text(rating_counts.index[i], v + 1000, f'{v:,}', 
                        ha='center', va='bottom', fontweight='bold', color='white')
        
        # Histogram with KDE
        sns.histplot(data=df_viz, x='rating', bins=10, kde=True, ax=axes[1], color='#46d369')
        axes[1].set_xlabel('Rating', fontsize=12, fontweight='bold', color='white')
        axes[1].set_ylabel('Density', fontsize=12, fontweight='bold', color='white')
        axes[1].set_title('Rating Distribution - Histogram with KDE', fontsize=14, fontweight='bold', pad=20, color='white')
        axes[1].grid(axis='y', alpha=0.3, color='gray')
        axes[1].set_facecolor('#1a1a1a')
        axes[1].tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Statistics
        with st.expander("📊 View Detailed Statistics"):
            for rating in sorted(df_viz['rating'].unique()):
                count = len(df_viz[df_viz['rating'] == rating])
                pct = (count / len(df_viz)) * 100
                st.write(f"⭐ Rating {rating}: {count:,} ({pct:.2f}%)")
        
        st.markdown("---")
        
        # 2. Top Movies
        st.markdown("### 🎬 2. Most Rated Movies")
        
        movie_counts = df_viz.groupby('movieId').size().reset_index(name='count')
        top_movies = movie_counts.nlargest(20, 'count')
        top_movies = top_movies.merge(movies_df_viz[['movieId', 'title']], on='movieId')
        
        fig = px.bar(top_movies, x='count', y='title', orientation='h',
                     title='Top 20 Most Rated Movies',
                     labels={'count': 'Number of Ratings', 'title': 'Movie Title'},
                     color='count',
                     color_continuous_scale='Reds')
        
        fig.update_layout(
            height=800,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_dark',
            paper_bgcolor='#141414',
            plot_bgcolor='#1a1a1a'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Genre Distribution
        st.markdown("### 🎭 3. Genre Distribution")
        
        all_genres = []
        for genres in movies_df_viz['genres'].dropna():
            if isinstance(genres, str) and genres != '(no genres listed)':
                all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts().head(15)
        
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                     title='Top 15 Popular Movie Genres',
                     labels={'x': 'Number of Movies', 'y': 'Genre'},
                     color=genre_counts.values,
                     color_continuous_scale='Blues')
        
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_dark',
            paper_bgcolor='#141414',
            plot_bgcolor='#1a1a1a'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 4. User Activity
        st.markdown("### 👥 4. User Activity")
        
        user_counts = df_viz.groupby('userId').size().reset_index(name='count')
        
        fig = px.histogram(user_counts, x='count', nbins=50,
                           title='Distribution of User Rating Counts',
                           labels={'count': 'Number of Ratings'},
                           color_discrete_sequence=['#e50914'])
        
        fig.update_layout(
            template='plotly_dark',
            showlegend=False,
            paper_bgcolor='#141414',
            plot_bgcolor='#1a1a1a'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # User statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Ratings/User", f"{user_counts['count'].mean():.1f}")
        with col2:
            st.metric("Median Ratings/User", f"{user_counts['count'].median():.0f}")
        with col3:
            st.metric("Max Ratings/User", f"{user_counts['count'].max()}")
        
        st.markdown("---")
        
        # 5. Rating Over Time
        st.markdown("### 📅 5. Rating Trends Over Time")
        
        df_time = df_viz.copy()
        df_time['date'] = pd.to_datetime(df_time['timestamp'], unit='s')
        df_time['year_month'] = df_time['date'].dt.to_period('M')
        
        monthly_ratings = df_time.groupby('year_month')['rating'].agg(['mean', 'count']).reset_index()
        monthly_ratings['year_month'] = monthly_ratings['year_month'].astype(str)
        
        # Only show last 50 months for better visualization
        monthly_ratings = monthly_ratings.tail(50)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_ratings['year_month'], y=monthly_ratings['mean'],
                       name="Avg Rating", line=dict(color='#46d369', width=2)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(x=monthly_ratings['year_month'], y=monthly_ratings['count'],
                   name="Count", marker_color='#e50914', opacity=0.3),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Month/Year")
        fig.update_yaxes(title_text="Average Rating", secondary_y=False)
        fig.update_yaxes(title_text="Number of Ratings", secondary_y=True)
        
        fig.update_layout(
            title_text="Rating Trends Over Time (Last 50 Months)",
            template='plotly_dark',
            height=500,
            paper_bgcolor='#141414',
            plot_bgcolor='#1a1a1a'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 6. Rating Heatmap
        st.markdown("### 🔥 6. Heatmap: Rating by Hour and Day")
        
        df_time['hour'] = df_time['date'].dt.hour
        df_time['dayofweek'] = df_time['date'].dt.dayofweek
        
        heatmap_data = df_time.pivot_table(
            values='rating', 
            index='dayofweek', 
            columns='hour', 
            aggfunc='mean'
        )
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data.index = [day_names[i] for i in heatmap_data.index]
        
        fig = px.imshow(heatmap_data,
                        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Rating"),
                        title="Heatmap: Average Rating by Hour and Day",
                        color_continuous_scale='RdYlGn',
                        aspect="auto")
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='#141414'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 7. Top Rated Movies (with minimum ratings threshold)
        st.markdown("### ⭐ 7. Top Rated Movies")
        
        min_ratings = st.slider("Minimum number of ratings:", 10, 100, 50, 10)
        
        movie_stats = df_viz.groupby('movieId').agg({
            'rating': ['mean', 'count'],
            'title': 'first'
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'avg_rating', 'count', 'title']
        top_rated = movie_stats[movie_stats['count'] >= min_ratings].nlargest(20, 'avg_rating')
        
        fig = px.bar(top_rated, x='avg_rating', y='title', orientation='h',
                     title=f'Top 20 Highest Rated Movies (≥{min_ratings} ratings)',
                     labels={'avg_rating': 'Average Rating', 'title': 'Movie Title'},
                     color='avg_rating',
                     color_continuous_scale='Greens',
                     hover_data=['count'])
        
        fig.update_layout(
            height=800,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_dark',
            paper_bgcolor='#141414',
            plot_bgcolor='#1a1a1a'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 8. Treemap - Tần suất Thể loại
        st.markdown("### 🎭 8. Treemap: Genre Frequency")
        
        # Create treemap data
        treemap_data = pd.DataFrame({
            'Genre': genre_counts.index,
            'Count': genre_counts.values
        })
        
        fig = px.treemap(
            treemap_data,
            path=['Genre'],
            values='Count',
            title='Movie Genre Frequency - Treemap',
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor='#141414'
        )
        
        fig.update_traces(
            textinfo='label+value',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 9. Heatmap - Ma trận Rating theo User và Movie
        st.markdown("### 🔥 9. Heatmap: Rating Matrix (Top Users & Movies)")
        
        # Get top users and movies for heatmap
        top_users_heatmap = df_viz['userId'].value_counts().head(50).index
        top_movies_heatmap = df_viz['movieId'].value_counts().head(30).index
        
        # Create pivot table
        heatmap_matrix = df_viz[
            (df_viz['userId'].isin(top_users_heatmap)) & 
            (df_viz['movieId'].isin(top_movies_heatmap))
        ].pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            aggfunc='mean'
        )
        
        # Create heatmap with matplotlib/seaborn
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('#141414')
        ax.set_facecolor('#1a1a1a')
        
        sns.heatmap(
            heatmap_matrix, 
            cmap='YlOrRd', 
            annot=False,
            cbar_kws={'label': 'Average Rating'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_title('Heatmap: Rating by User and Movie (Top 50 Users & Top 30 Movies)', 
                     fontsize=14, fontweight='bold', pad=20, color='white')
        ax.set_xlabel('Movie ID', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('User ID', fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info(f"Heatmap displays {len(top_users_heatmap)} users and {len(top_movies_heatmap)} movies")
        
        st.markdown("---")
        
        # 10. WordCloud - Từ khóa từ Tiêu đề Phim
        st.markdown("### ☁️ 10. WordCloud: Keywords from Movie Titles")
        
        try:
            from wordcloud import WordCloud
            
            # Combine all movie titles
            all_titles = ' '.join(movies_df_viz['title'].dropna().astype(str))
            
            # Create WordCloud
            wordcloud = WordCloud(
                width=1600,
                height=800,
                background_color='#141414',
                colormap='Set2',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(all_titles)
            
            # Display WordCloud
            fig, ax = plt.subplots(figsize=(16, 8))
            fig.patch.set_facecolor('#141414')
            ax.set_facecolor('#141414')
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('WordCloud: Popular Keywords in Movie Titles', 
                        fontsize=16, fontweight='bold', pad=20, color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except ImportError:
            st.warning("⚠️ Need to install 'wordcloud' library to display this chart.")
            st.code("pip install wordcloud", language="bash")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please ensure data/ratings.csv and data/movies.csv files exist.")

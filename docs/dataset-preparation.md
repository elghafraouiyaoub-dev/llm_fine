# Dataset Preparation

## Overview

This document outlines the data preprocessing and feature engineering pipeline for the MovieLens 25M dataset, which forms the foundation of our LLM-based recommendation system.

## Dataset Description

### MovieLens 25M Dataset
- **Size**: 25 million ratings from 162,000 users on 62,000 movies
- **Time Period**: January 1995 to November 2019
- **Rating Scale**: 0.5 to 5.0 stars (0.5 increments)
- **Additional Data**: Movie metadata, user tags, genome scores

### Data Files Structure
```
ml-25m/
├── ratings.csv          # User ratings (userId, movieId, rating, timestamp)
├── movies.csv           # Movie metadata (movieId, title, genres)
├── tags.csv             # User-generated tags (userId, movieId, tag, timestamp)
├── links.csv            # External links (movieId, imdbId, tmdbId)
├── genome-scores.csv    # Movie-tag relevance scores
└── genome-tags.csv      # Tag descriptions
```

## Data Preprocessing Pipeline

### 1. Data Loading and Validation

```python
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
    
    def load_ratings(self) -> pd.DataFrame:
        """Load and validate ratings data."""
        ratings = pd.read_csv(f"{self.data_path}/ratings.csv")
        
        # Validation checks
        assert ratings['rating'].between(0.5, 5.0).all(), "Invalid rating values"
        assert ratings['userId'].notna().all(), "Missing user IDs"
        assert ratings['movieId'].notna().all(), "Missing movie IDs"
        
        # Convert timestamp to datetime
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        return ratings
    
    def load_movies(self) -> pd.DataFrame:
        """Load and clean movie metadata."""
        movies = pd.read_csv(f"{self.data_path}/movies.csv")
        
        # Extract year from title
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        
        # Clean title (remove year)
        movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Process genres
        movies['genres_list'] = movies['genres'].str.split('|')
        movies['num_genres'] = movies['genres_list'].str.len()
        
        return movies
```

### 2. Feature Engineering

#### User Features
```python
def create_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive user features."""
    user_stats = ratings.groupby('userId').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max'],
        'timestamp': ['min', 'max']
    }).round(3)
    
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
    
    # Additional features
    user_stats['rating_range'] = user_stats['rating_max'] - user_stats['rating_min']
    user_stats['days_active'] = (user_stats['timestamp_max'] - user_stats['timestamp_min']).dt.days
    user_stats['ratings_per_day'] = user_stats['rating_count'] / (user_stats['days_active'] + 1)
    
    # User behavior categories
    user_stats['user_type'] = pd.cut(
        user_stats['rating_count'], 
        bins=[0, 20, 100, 500, float('inf')], 
        labels=['casual', 'regular', 'active', 'power']
    )
    
    return user_stats
```

#### Movie Features
```python
def create_movie_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive movie features."""
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['count', 'mean', 'std'],
        'userId': 'nunique'
    }).round(3)
    
    movie_stats.columns = ['_'.join(col).strip() for col in movie_stats.columns]
    
    # Merge with movie metadata
    movie_features = movies.merge(movie_stats, on='movieId', how='left')
    
    # Popularity and quality metrics
    movie_features['popularity_score'] = np.log1p(movie_features['rating_count'])
    movie_features['quality_score'] = movie_features['rating_mean'] * np.log1p(movie_features['rating_count'])
    
    # Genre-based features
    genre_dummies = movie_features['genres'].str.get_dummies(sep='|')
    movie_features = pd.concat([movie_features, genre_dummies], axis=1)
    
    return movie_features
```

### 3. Text Processing for LLM Training

#### Content Preparation
```python
def prepare_llm_training_data(ratings: pd.DataFrame, movies: pd.DataFrame, 
                            tags: pd.DataFrame) -> pd.DataFrame:
    """Prepare text data for LLM fine-tuning."""
    
    # Create user-movie interaction summaries
    user_movie_data = []
    
    for user_id in ratings['userId'].unique()[:1000]:  # Sample for efficiency
        user_ratings = ratings[ratings['userId'] == user_id].sort_values('timestamp')
        
        # Get user's rating history
        rated_movies = user_ratings.merge(movies, on='movieId')
        
        # Create training examples
        for i in range(len(rated_movies) - 1):
            # Historical context
            history = rated_movies.iloc[:i+1]
            target = rated_movies.iloc[i+1]
            
            # Format as conversation
            context = create_recommendation_context(history, target)
            user_movie_data.append(context)
    
    return pd.DataFrame(user_movie_data)

def create_recommendation_context(history: pd.DataFrame, target: pd.Series) -> Dict[str, str]:
    """Create conversational context for LLM training."""
    
    # User preferences summary
    liked_movies = history[history['rating'] >= 4.0]['clean_title'].tolist()
    disliked_movies = history[history['rating'] <= 2.0]['clean_title'].tolist()
    
    # Genre preferences
    genre_ratings = {}
    for _, row in history.iterrows():
        for genre in row['genres_list']:
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(row['rating'])
    
    preferred_genres = [genre for genre, ratings in genre_ratings.items() 
                       if np.mean(ratings) >= 4.0]
    
    # Create prompt
    prompt = f"""Based on a user's movie preferences, recommend whether they would like a specific movie.

User's liked movies: {', '.join(liked_movies[:5])}
User's preferred genres: {', '.join(preferred_genres[:3])}
Movie to evaluate: {target['clean_title']} ({target['genres']})

Would this user like this movie? Provide a rating prediction (1-5) and explanation."""
    
    response = f"Rating: {target['rating']}/5\nExplanation: Based on the user's preferences for {', '.join(preferred_genres[:2])}, this movie aligns well with their taste."
    
    return {
        'prompt': prompt,
        'response': response,
        'user_id': target['userId'],
        'movie_id': target['movieId']
    }
```

### 4. Data Splitting Strategy

```python
def create_temporal_splits(ratings: pd.DataFrame, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create temporal train/validation/test splits."""
    
    # Sort by timestamp
    ratings_sorted = ratings.sort_values('timestamp')
    
    # Calculate split points
    n_total = len(ratings_sorted)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    
    train_data = ratings_sorted.iloc[:train_end]
    val_data = ratings_sorted.iloc[train_end:val_end]
    test_data = ratings_sorted.iloc[val_end:]
    
    return train_data, val_data, test_data

def create_user_based_splits(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create user-based train/test splits for each user."""
    
    train_data = []
    test_data = []
    
    for user_id in ratings['userId'].unique():
        user_ratings = ratings[ratings['userId'] == user_id].sort_values('timestamp')
        
        if len(user_ratings) >= 5:  # Minimum ratings threshold
            # Use last 20% of ratings for testing
            split_point = int(len(user_ratings) * 0.8)
            train_data.append(user_ratings.iloc[:split_point])
            test_data.append(user_ratings.iloc[split_point:])
    
    return pd.concat(train_data), pd.concat(test_data)
```

### 5. Data Quality Checks

```python
def validate_data_quality(ratings: pd.DataFrame, movies: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality validation."""
    
    quality_report = {
        'ratings': {
            'total_ratings': len(ratings),
            'unique_users': ratings['userId'].nunique(),
            'unique_movies': ratings['movieId'].nunique(),
            'rating_distribution': ratings['rating'].value_counts().to_dict(),
            'missing_values': ratings.isnull().sum().to_dict(),
            'duplicate_ratings': ratings.duplicated(['userId', 'movieId']).sum()
        },
        'movies': {
            'total_movies': len(movies),
            'missing_genres': movies['genres'].isnull().sum(),
            'missing_years': movies['year'].isnull().sum(),
            'genre_distribution': movies['genres'].value_counts().head(10).to_dict()
        }
    }
    
    # Data consistency checks
    orphaned_ratings = set(ratings['movieId']) - set(movies['movieId'])
    quality_report['consistency'] = {
        'orphaned_ratings': len(orphaned_ratings),
        'coverage': len(set(ratings['movieId']) & set(movies['movieId'])) / len(movies)
    }
    
    return quality_report
```

### 6. Feature Store Integration

```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64, String

def setup_feature_store():
    """Configure Feast feature store for serving features."""
    
    # Define entities
    user_entity = Entity(name="user_id", value_type=Int64)
    movie_entity = Entity(name="movie_id", value_type=Int64)
    
    # Define feature views
    user_features_view = FeatureView(
        name="user_features",
        entities=[user_entity],
        schema=[
            Field(name="avg_rating", dtype=Float32),
            Field(name="rating_count", dtype=Int64),
            Field(name="user_type", dtype=String),
        ],
        source=user_features_source,  # Define data source
        ttl=timedelta(days=1)
    )
    
    movie_features_view = FeatureView(
        name="movie_features", 
        entities=[movie_entity],
        schema=[
            Field(name="avg_rating", dtype=Float32),
            Field(name="popularity_score", dtype=Float32),
            Field(name="genres", dtype=String),
        ],
        source=movie_features_source,
        ttl=timedelta(days=7)
    )
    
    return FeatureStore(repo_path=".")
```

## Data Pipeline Orchestration

### Automated Pipeline
```python
def run_data_pipeline(data_path: str, output_path: str):
    """Execute complete data preprocessing pipeline."""
    
    # 1. Load raw data
    loader = DataLoader(data_path)
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    tags = loader.load_tags()
    
    # 2. Feature engineering
    user_features = create_user_features(ratings)
    movie_features = create_movie_features(ratings, movies)
    
    # 3. Prepare LLM training data
    llm_data = prepare_llm_training_data(ratings, movies, tags)
    
    # 4. Create data splits
    train_ratings, val_ratings, test_ratings = create_temporal_splits(ratings)
    
    # 5. Quality validation
    quality_report = validate_data_quality(ratings, movies)
    
    # 6. Save processed data
    save_processed_data(output_path, {
        'train_ratings': train_ratings,
        'val_ratings': val_ratings, 
        'test_ratings': test_ratings,
        'user_features': user_features,
        'movie_features': movie_features,
        'llm_training_data': llm_data,
        'quality_report': quality_report
    })
    
    return quality_report
```

## Next Steps

1. **Data Download**: Implement automated MovieLens dataset download
2. **Pipeline Automation**: Set up Apache Airflow for pipeline orchestration
3. **Feature Store**: Deploy Feast for real-time feature serving
4. **Data Monitoring**: Implement data drift detection with Evidently
5. **Incremental Processing**: Add support for incremental data updates

This preprocessing pipeline ensures high-quality, well-structured data for training our LLM-based recommendation system.

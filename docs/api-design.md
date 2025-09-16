# API Design

## Overview

This document outlines the REST API design for our LLM-based movie recommendation system, including endpoint specifications, authentication patterns, rate limiting, and integration strategies.

## API Architecture

### Design Principles
- **RESTful**: Follow REST conventions for resource-based URLs
- **Stateless**: Each request contains all necessary information
- **Cacheable**: Responses are cacheable where appropriate
- **Versioned**: API versioning for backward compatibility
- **Secure**: Authentication and authorization for all endpoints

### Technology Stack
- **Framework**: FastAPI with async support
- **Authentication**: JWT tokens with refresh mechanism
- **Validation**: Pydantic models for request/response validation
- **Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Rate Limiting**: Redis-based rate limiting
- **Caching**: Multi-level caching strategy

## API Endpoints

### Authentication Endpoints

#### POST /auth/login
```python
# Request
{
    "username": "user@example.com",
    "password": "secure_password"
}

# Response
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
        "id": 123,
        "username": "user@example.com",
        "preferences": {...}
    }
}
```

#### POST /auth/refresh
```python
# Request
{
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}

# Response
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600
}
```

### User Management Endpoints

#### GET /users/{user_id}/profile
```python
# Response
{
    "user_id": 123,
    "username": "user@example.com",
    "preferences": {
        "favorite_genres": ["Action", "Sci-Fi"],
        "disliked_genres": ["Horror"],
        "rating_style": "generous",
        "discovery_preference": "popular"
    },
    "statistics": {
        "total_ratings": 245,
        "average_rating": 3.8,
        "genres_watched": 15,
        "last_activity": "2024-01-15T10:30:00Z"
    }
}
```

#### PUT /users/{user_id}/preferences
```python
# Request
{
    "favorite_genres": ["Action", "Comedy"],
    "disliked_genres": ["Horror", "Documentary"],
    "discovery_preference": "diverse"
}

# Response
{
    "message": "Preferences updated successfully",
    "updated_at": "2024-01-15T10:30:00Z"
}
```

### Recommendation Endpoints

#### POST /recommendations
```python
# Request
{
    "user_id": 123,
    "num_recommendations": 10,
    "filters": {
        "genres": ["Action", "Sci-Fi"],
        "min_year": 2010,
        "max_year": 2024,
        "min_rating": 7.0,
        "exclude_watched": true
    },
    "explanation": true,
    "diversity_factor": 0.7
}

# Response
{
    "recommendations": [
        {
            "movie_id": 456,
            "title": "Dune: Part Two",
            "year": 2024,
            "genres": ["Action", "Adventure", "Drama"],
            "predicted_rating": 4.2,
            "confidence": 0.85,
            "poster_url": "https://image.tmdb.org/t/p/w500/...",
            "explanation": "Based on your love for epic sci-fi films like Blade Runner 2049 and your high ratings for Denis Villeneuve movies, you'll likely enjoy this continuation of the Dune saga.",
            "similarity_reasons": [
                "Director: Denis Villeneuve (you rated Arrival 5/5)",
                "Genre: Epic Sci-Fi (matches your preferences)",
                "Visual Style: Cinematographic excellence"
            ]
        }
    ],
    "metadata": {
        "algorithm_version": "v2.1",
        "generation_time_ms": 245,
        "cache_hit": false,
        "diversity_score": 0.73
    }
}
```

#### GET /recommendations/{user_id}/history
```python
# Response
{
    "recommendations": [
        {
            "recommendation_id": "rec_789",
            "timestamp": "2024-01-15T10:30:00Z",
            "movies": [...],
            "user_feedback": {
                "clicked": [456, 789],
                "rated": [{"movie_id": 456, "rating": 4}],
                "dismissed": [123]
            }
        }
    ],
    "pagination": {
        "page": 1,
        "per_page": 20,
        "total": 150,
        "has_next": true
    }
}
```

### Movie Information Endpoints

#### GET /movies/{movie_id}
```python
# Response
{
    "movie_id": 456,
    "title": "Dune: Part Two",
    "original_title": "Dune: Part Two",
    "year": 2024,
    "runtime": 166,
    "genres": ["Action", "Adventure", "Drama"],
    "director": "Denis Villeneuve",
    "cast": ["Timothée Chalamet", "Zendaya", "Rebecca Ferguson"],
    "plot": "Paul Atreides unites with Chani and the Fremen...",
    "ratings": {
        "imdb": 8.8,
        "tmdb": 8.4,
        "system_average": 4.2,
        "system_count": 15420
    },
    "metadata": {
        "poster_url": "https://image.tmdb.org/t/p/w500/...",
        "backdrop_url": "https://image.tmdb.org/t/p/original/...",
        "trailer_url": "https://youtube.com/watch?v=...",
        "imdb_id": "tt15239678",
        "tmdb_id": 693134
    }
}
```

#### GET /movies/search
```python
# Query Parameters: ?q=dune&year=2024&genre=sci-fi&limit=10

# Response
{
    "results": [
        {
            "movie_id": 456,
            "title": "Dune: Part Two",
            "year": 2024,
            "genres": ["Action", "Adventure", "Drama"],
            "rating": 8.8,
            "poster_url": "https://image.tmdb.org/t/p/w500/..."
        }
    ],
    "pagination": {
        "page": 1,
        "per_page": 10,
        "total": 25,
        "has_next": true
    }
}
```

### Rating and Feedback Endpoints

#### POST /ratings
```python
# Request
{
    "user_id": 123,
    "movie_id": 456,
    "rating": 4.5,
    "review": "Absolutely stunning visuals and compelling story continuation.",
    "tags": ["epic", "visually-stunning", "great-sequel"]
}

# Response
{
    "rating_id": "rating_789",
    "message": "Rating submitted successfully",
    "updated_recommendation_model": true,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /feedback
```python
# Request
{
    "recommendation_id": "rec_789",
    "user_id": 123,
    "feedback_type": "click",
    "movie_id": 456,
    "timestamp": "2024-01-15T10:30:00Z",
    "context": {
        "position": 2,
        "page": 1,
        "session_id": "sess_abc123"
    }
}

# Response
{
    "message": "Feedback recorded successfully",
    "feedback_id": "fb_456"
}
```

## FastAPI Implementation

### Main Application Structure
```python
# src/api/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
import uvicorn

from .routers import auth, users, recommendations, movies, ratings
from .middleware import RateLimitMiddleware, LoggingMiddleware
from .dependencies import get_current_user, get_db
from .config import settings

app = FastAPI(
    title="Movie Recommendation API",
    description="LLM-powered movie recommendation system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

# Security
security = HTTPBearer()

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(movies.router, prefix="/movies", tags=["movies"])
app.include_router(ratings.router, prefix="/ratings", tags=["ratings"])

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": "2024-01-15T10:30:00Z"
    }

@app.get("/ready")
async def readiness_check(db=Depends(get_db)):
    """Readiness check endpoint."""
    try:
        # Check database connection
        await db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
```

### Recommendation Router
```python
# src/api/routers/recommendations.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
import asyncio

from ..models import RecommendationRequest, RecommendationResponse
from ..dependencies import get_current_user, get_recommendation_service
from ..services import RecommendationService, CacheService

router = APIRouter()

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Generate personalized movie recommendations."""
    
    try:
        # Check cache first
        cache_key = f"rec_{request.user_id}_{hash(str(request.dict()))}"
        cached_result = await rec_service.get_cached_recommendations(cache_key)
        
        if cached_result:
            return cached_result
        
        # Generate recommendations
        recommendations = await rec_service.generate_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            filters=request.filters,
            explanation=request.explanation,
            diversity_factor=request.diversity_factor
        )
        
        # Cache result
        background_tasks.add_task(
            rec_service.cache_recommendations,
            cache_key,
            recommendations
        )
        
        # Log recommendation request
        background_tasks.add_task(
            rec_service.log_recommendation_request,
            request,
            recommendations
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@router.get("/{user_id}/similar-users")
async def get_similar_users(
    user_id: int,
    limit: int = 10,
    current_user=Depends(get_current_user),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Find users with similar preferences."""
    
    similar_users = await rec_service.find_similar_users(user_id, limit)
    return {"similar_users": similar_users}
```

### Pydantic Models
```python
# src/api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class GenreEnum(str, Enum):
    ACTION = "Action"
    ADVENTURE = "Adventure"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    HORROR = "Horror"
    SCI_FI = "Sci-Fi"
    THRILLER = "Thriller"

class RecommendationFilters(BaseModel):
    genres: Optional[List[GenreEnum]] = None
    min_year: Optional[int] = Field(None, ge=1900, le=2030)
    max_year: Optional[int] = Field(None, ge=1900, le=2030)
    min_rating: Optional[float] = Field(None, ge=0.0, le=10.0)
    max_rating: Optional[float] = Field(None, ge=0.0, le=10.0)
    exclude_watched: bool = True
    
    @validator('max_year')
    def validate_year_range(cls, v, values):
        if 'min_year' in values and values['min_year'] and v:
            if v < values['min_year']:
                raise ValueError('max_year must be greater than min_year')
        return v

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    num_recommendations: int = Field(10, ge=1, le=100)
    filters: Optional[RecommendationFilters] = None
    explanation: bool = True
    diversity_factor: float = Field(0.5, ge=0.0, le=1.0)

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: List[str]
    predicted_rating: float = Field(..., ge=0.0, le=5.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    poster_url: Optional[str] = None
    explanation: Optional[str] = None
    similarity_reasons: Optional[List[str]] = None

class RecommendationMetadata(BaseModel):
    algorithm_version: str
    generation_time_ms: int
    cache_hit: bool
    diversity_score: float

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    metadata: RecommendationMetadata
```

## Authentication & Security

### JWT Authentication
```python
# src/api/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import bcrypt

from .config import settings
from .models import User

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    
    return user
```

### Rate Limiting
```python
# src/api/middleware.py
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis
import time
import json

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = "redis://localhost:6379"):
        super().__init__(app)
        self.redis_client = redis.from_url(redis_url)
        self.rate_limits = {
            "/recommendations": {"requests": 10, "window": 60},  # 10 requests per minute
            "/movies/search": {"requests": 100, "window": 60},   # 100 requests per minute
            "default": {"requests": 1000, "window": 60}          # 1000 requests per minute
        }
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Get rate limit for endpoint
        endpoint = request.url.path
        rate_limit = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        # Check rate limit
        key = f"rate_limit:{client_ip}:{endpoint}"
        current_requests = self.redis_client.get(key)
        
        if current_requests is None:
            # First request in window
            self.redis_client.setex(key, rate_limit["window"], 1)
        else:
            current_requests = int(current_requests)
            if current_requests >= rate_limit["requests"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            else:
                self.redis_client.incr(key)
        
        response = await call_next(request)
        return response
```

## Caching Strategy

### Multi-Level Caching
```python
# src/api/cache.py
from typing import Any, Optional
import json
import hashlib
from datetime import timedelta
import redis
from functools import wraps

class CacheService:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception:
            return False
    
    def cached(self, ttl: int = None, prefix: str = "cache"):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.cache_key(f"{prefix}:{func.__name__}", *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
```

## API Testing

### Integration Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestRecommendationAPI:
    def test_get_recommendations_success(self):
        """Test successful recommendation request."""
        payload = {
            "user_id": 1,
            "num_recommendations": 5,
            "explanation": True
        }
        
        response = client.post("/recommendations/", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 5
        assert "metadata" in data
    
    def test_get_recommendations_with_filters(self):
        """Test recommendations with filters."""
        payload = {
            "user_id": 1,
            "num_recommendations": 10,
            "filters": {
                "genres": ["Action", "Sci-Fi"],
                "min_year": 2020,
                "exclude_watched": True
            }
        }
        
        response = client.post("/recommendations/", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify filters are applied
        for rec in data["recommendations"]:
            assert rec["year"] >= 2020
            assert any(genre in ["Action", "Sci-Fi"] for genre in rec["genres"])
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        for _ in range(15):  # Exceed rate limit of 10/minute
            response = client.post("/recommendations/", json={"user_id": 1})
        
        # Should get rate limited
        assert response.status_code == 429
```

## Next Steps

1. **GraphQL Support**: Add GraphQL endpoint for flexible queries
2. **WebSocket Integration**: Real-time recommendation updates
3. **API Gateway**: Implement Kong or similar for advanced routing
4. **Monitoring Integration**: Add detailed API metrics and tracing
5. **Mobile SDK**: Create SDKs for mobile app integration

This API design provides a comprehensive, scalable, and secure foundation for our movie recommendation system.

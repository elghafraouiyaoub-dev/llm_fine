"""
Configuration settings for the movie recommendation API.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Movie Recommendation API"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/recommendations"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Ollama
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:7b"
    OLLAMA_TIMEOUT: int = 60
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "movie-recommendations"
    
    # Weights & Biases
    WANDB_PROJECT: str = "movie-recommendation-system"
    WANDB_ENTITY: Optional[str] = None
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080"
    ]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Recommendation Settings
    DEFAULT_RECOMMENDATIONS_COUNT: int = 10
    MAX_RECOMMENDATIONS_COUNT: int = 100
    RECOMMENDATION_CACHE_TTL: int = 1800  # 30 minutes
    
    # Model Settings
    MODEL_BATCH_SIZE: int = 32
    MODEL_MAX_LENGTH: int = 2048
    MODEL_TEMPERATURE: float = 0.7
    MODEL_TOP_P: float = 0.9
    MODEL_TOP_K: int = 40
    
    # Data Processing
    DATA_PATH: str = "./data"
    MODELS_PATH: str = "./models"
    LOGS_PATH: str = "./logs"
    
    # Monitoring
    PROMETHEUS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    
    # Feature Flags
    ENABLE_EXPLANATIONS: bool = True
    ENABLE_DIVERSITY_BOOST: bool = True
    ENABLE_COLD_START_HANDLING: bool = True
    ENABLE_A_B_TESTING: bool = False
    
    # External APIs
    TMDB_API_KEY: Optional[str] = None
    IMDB_API_KEY: Optional[str] = None
    
    # Security
    BCRYPT_ROUNDS: int = 12
    JWT_ALGORITHM: str = "HS256"
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".parquet"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        for path in [self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT.lower() == "production"

# Create global settings instance
settings = Settings()

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": f"{settings.LOGS_PATH}/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "sqlalchemy": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
    },
}

# Model configuration
MODEL_CONFIG = {
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "fp16": True,
        "dataloader_num_workers": 4
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": True,
        "pad_token_id": 0,
        "eos_token_id": 2
    }
}

# Cache configuration
CACHE_CONFIG = {
    "recommendations": {
        "ttl": 1800,  # 30 minutes
        "max_size": 10000
    },
    "user_profiles": {
        "ttl": 3600,  # 1 hour
        "max_size": 50000
    },
    "movie_features": {
        "ttl": 86400,  # 24 hours
        "max_size": 100000
    },
    "model_predictions": {
        "ttl": 7200,  # 2 hours
        "max_size": 20000
    }
}

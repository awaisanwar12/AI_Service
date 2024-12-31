from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Telegram settings
    TELEGRAM_TOKEN: str
    X_AI_API_KEY: str
    PORT: int = 8000
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./app.db"
    
    
    
    # Pinecone settings
    PINECONE_API_KEY: str
    PINECONE_ENV: str
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_URL: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 
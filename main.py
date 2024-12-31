import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
# from app.services.telegram_bot import bot_service  # Commented out telegram bot import
from app.core.config import get_settings
import logging
from app.middleware.rate_limiter import RateLimiterMiddleware  # Import the middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="AI Assistant API",
    description="FastAPI backend for X AI integration",  # Updated description
    version="1.0.0"
)

# Add rate limiting middleware
app.add_middleware(RateLimiterMiddleware, max_requests=5, window_seconds=60)  # 5 requests per minute

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Comment out Telegram bot start
        # asyncio.create_task(bot_service.start())
        # logger.info("Telegram bot started successfully")
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Comment out Telegram bot stop
        # await bot_service.stop()
        # logger.info("Telegram bot stopped successfully")
        logger.info("Application stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping application: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint for health check
    """
    return {"status": "Server is running"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    ) 
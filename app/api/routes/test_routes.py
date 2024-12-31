from fastapi import APIRouter
from app.services.redis_service import redis_service

router = APIRouter()

@router.get("/test-redis")
async def test_redis():
    try:
        # Set a test value
        await redis_service.set_cache("test_key", {"message": "Hello, Redis!"}, expire_seconds=60)
        # Retrieve the test value
        value = await redis_service.get_cache("test_key")
        return {"status": "success", "value": value}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/test")
async def test():
    return {"status": "success", "message": "Test endpoint is working"}

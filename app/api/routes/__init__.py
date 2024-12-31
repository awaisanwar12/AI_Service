from fastapi import APIRouter
from app.api.routes.chat import router as chat_router
from app.api.routes.test_routes import router as test_router

router = APIRouter()
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(test_router, prefix="/test", tags=["test"]) 
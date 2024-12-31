from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.redis_service import redis_service
from app.services.rag_service import rag_service
from app.models.database import SessionLocal, Conversation
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Use SessionLocal to create a new session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PromptRequest(BaseModel):
    prompt: str

class ContentResponse(BaseModel):
    text: str
    image_url: str | None = None

@router.post("/generate", response_model=ContentResponse)
async def generate_chat_response(request: PromptRequest):
    """
    Generate a response using X AI API with Redis cache (1 minute) and Pinecone fallback
    """
    try:
        # Initialize Redis connection if needed
        await redis_service.initialize_redis()

        # Check Redis cache first (fast in-memory cache, 1 minute TTL)
        logger.info("Checking Redis cache...")
        cached_response = await redis_service.get_cached_response(request.prompt)
        if cached_response:
            logger.info("Using cached response from Redis (1 minute TTL)")
            return ContentResponse(
                text=cached_response["text"],
                image_url=cached_response.get("image_url")
            )

        # If Redis cache miss, check Pinecone for similar questions
        logger.info("Redis cache miss (or expired), checking Pinecone for similar responses...")
        similar_responses = await rag_service.search_similar(request.prompt, limit=3)  # Get top 3 matches
        
        # Log all matches for debugging
        for idx, resp in enumerate(similar_responses):
            logger.info(f"Match {idx + 1}:")
            logger.info(f"  Score: {resp['score']}")
            logger.info(f"  Stored message: {resp['message']}")
            logger.info(f"  Response: {resp['response'][:100]}...")

        # Check for good matches (lowered threshold to 0.3)
        if similar_responses and similar_responses[0]["score"] > 0.3:
            best_match = similar_responses[0]
            logger.info(f"Found good match in Pinecone with score {best_match['score']}")
            logger.info(f"Original query: '{request.prompt}'")
            logger.info(f"Matched query: '{best_match['message']}'")
            
            pinecone_response = {
                "text": best_match["response"],
                "image_url": None
            }
            # Cache the Pinecone response in Redis for future fast access (1 minute)
            logger.info("Caching Pinecone response in Redis for 1 minute...")
            await redis_service.cache_response(request.prompt, pinecone_response)
            return ContentResponse(**pinecone_response)

        # If no similar response found, generate new one
        logger.info("No similar response found, generating new response from X AI...")
        result = await rag_service.process_message(request.prompt, "")

        # Store in SQLite for persistence
        session = SessionLocal()
        new_conversation = Conversation(
            chat_id="default",
            user_message=request.prompt,
            bot_response=result["text"]
        )
        session.add(new_conversation)
        session.commit()
        session.close()

        return ContentResponse(
            text=result["text"],
            image_url=result.get("image_url")
        )
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/train")
async def train_vectors():
    """
    Train the vector database with sample data
    """
    try:
        from app.utils.train_vectors import train_vector_database
        await train_vector_database()
        return {"status": "success", "message": "Vector database trained successfully"}
    except Exception as e:
        logger.error(f"Error training vectors: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to train vector database"
        )

@router.get("/test-pinecone")
async def test_pinecone():
    """
    Test Pinecone operations
    """
    try:
        # Get index stats
        stats = rag_service.index.describe_index_stats()
        
        # Test storing a sample message
        test_message = "This is a test message"
        test_response = "This is a test response"
        await rag_service.process_message(test_message, test_response)
        
        # Get updated stats
        updated_stats = rag_service.index.describe_index_stats()
        
        return {
            "status": "success",
            "initial_stats": stats,
            "updated_stats": updated_stats,
            "message": "Pinecone test completed successfully"
        }
    except Exception as e:
        logger.error(f"Error testing Pinecone: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test Pinecone: {str(e)}"
        ) 
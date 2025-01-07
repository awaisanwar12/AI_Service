import os
from typing import List, Dict, Optional
import httpx
from pinecone import Pinecone, Index, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from app.core.config import get_settings
from app.services.redis_service import redis_service
import logging
import json
import hashlib
import datetime
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# Disable noisy logs
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Configure service-specific logging with clear formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create loggers with custom names for clarity
logger = logging.getLogger('RAG')
redis_logger = logging.getLogger('REDIS')
pinecone_logger = logging.getLogger('PINECONE')
xai_logger = logging.getLogger('X-AI')

settings = get_settings()

class RAGService:
    def __init__(self):
        try:
            logger.info("Starting RAG service initialization...")
            start_time = time.time()
            
            # Initialize the embedding model
            logger.info("Loading E5 large embedding model...")
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            logger.info("Embedding model loaded successfully")
            
            # Create an instance of Pinecone
            pinecone_logger.info(f"Connecting to Pinecone [API: {os.environ.get('PINECONE_API_KEY')[:5]}...]")
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            pinecone_logger.info("Pinecone client initialized")
            
            # Connect to the index
            pinecone_logger.info("Connecting to index: awais-test")
            self.index = self.pc.Index(
                "awais-test",
                host="https://awais-test-8unj2wq.svc.aped-4627-b74a.pinecone.io"
            )
            
            # Get index info
            index_info = self.index.describe_index_stats()
            stats_dict = {
                "dimension": index_info.dimension,
                "total_vectors": index_info.total_vector_count,
                "namespaces": {ns: {"count": info.vector_count} for ns, info in index_info.namespaces.items()}
            }
            pinecone_logger.info(f"Connected to Pinecone index | Stats: {json.dumps(stats_dict)}")
            
            end_time = time.time()
            logger.info(f"RAG service initialized in {end_time - start_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using multilingual-e5-large model"""
        try:
            # Normalize text
            text = ' '.join(text.lower().split())
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def is_gaming_related(self, text: str) -> bool:
        """Check if the text is related to gaming"""
        gaming_keywords = {
            # General gaming terms
            'game', 'gaming', 'play', 'player', 'gamer', 'gameplay', 'playthrough',
            # Platforms
            'pc', 'console', 'playstation', 'ps4', 'ps5', 'xbox', 'nintendo', 'switch',
            # Game types
            'rpg', 'fps', 'mmorpg', 'strategy', 'puzzle', 'arcade', 'simulator',
            # Gaming actions
            'level', 'score', 'achievement', 'quest', 'mission', 'boss', 'character',
            # Popular games
            'minecraft', 'fortnite', 'cod', 'gta', 'league of legends', 'valorant',
            # Gaming hardware
            'controller', 'keyboard', 'mouse', 'headset', 'gpu', 'graphics card',
            # Gaming terms
            'spawn', 'respawn', 'loot', 'inventory', 'skill', 'weapon', 'map',
            # Esports
            'tournament', 'competitive', 'esports', 'stream', 'twitch'
        }
        
        # Convert text to lowercase for matching
        text = text.lower()
        
        # Check if any gaming keyword is in the text
        return any(keyword in text for keyword in gaming_keywords)

    async def process_message(self, message: str, response: str):
        """Process message with Redis -> Pinecone -> X AI fallback"""
        process_start = time.time()
        try:
            logger.info("=" * 80)
            logger.info(f"NEW MESSAGE: '{message[:100]}'")
            
            # Check if message is gaming related
            if not self.is_gaming_related(message):
                logger.info("MESSAGE REJECTED: Not gaming related")
                logger.info("=" * 80)
                return {
                    "text": "I can only help with gaming-related questions. Please ask me about games, gaming, or anything related to gaming!",
                    "image_url": None
                }
            
            logger.info("MESSAGE ACCEPTED: Gaming related")
            
            # 1. Check Redis cache
            redis_logger.info("Step 1: Checking Redis cache...")
            cache_key = f"response:{hashlib.md5(message.encode()).hexdigest()}"
            cached_response = await redis_service.get_cache(cache_key)
            
            if cached_response:
                redis_logger.info("SUCCESS: Found response in Redis cache")
                logger.info("-" * 40)
                logger.info("RESPONSE SOURCE: Redis Cache")
                logger.info(f"RESPONSE PREVIEW: {cached_response['text'][:100]}...")
                logger.info(f"RESPONSE LENGTH: {len(cached_response['text'])} characters")
                logger.info(f"TOTAL TIME: {time.time() - process_start:.1f}s")
                logger.info("=" * 80)
                return cached_response
            
            redis_logger.info("INFO: No cache found in Redis")
            
            # 2. Search Pinecone
            pinecone_logger.info("Step 2: Searching Pinecone database...")
            similar_responses = await self.search_similar(message, limit=3)  # Get top 3 to log alternatives
            
            # Log all potential matches for debugging
            if similar_responses:
                pinecone_logger.info("Found potential matches:")
                for idx, resp in enumerate(similar_responses):
                    pinecone_logger.info(f"Match {idx + 1}:")
                    pinecone_logger.info(f"  Score: {resp['score']:.3f}")
                    pinecone_logger.info(f"  Query: '{message}'")
                    pinecone_logger.info(f"  Matched: '{resp['message']}'")
            
            # Check for good match - increased threshold to 0.85 for better accuracy
            if similar_responses and similar_responses[0]["score"] > 0.85:
                pinecone_logger.info(f"SUCCESS: Found similar message in Pinecone [similarity: {similar_responses[0]['score']:.3f}]")
                pinecone_logger.info("-" * 40)
                pinecone_logger.info("MATCH DETAILS:")
                pinecone_logger.info(f"User Query: '{message}'")
                pinecone_logger.info(f"Matched Query: '{similar_responses[0]['message']}'")
                pinecone_logger.info(f"Similarity Score: {similar_responses[0]['score']:.3f}")
                pinecone_logger.info("-" * 40)
                
                existing_response = {
                    "text": similar_responses[0]["response"],
                    "image_url": None
                }
                
                # Cache for future use
                redis_logger.info("INFO: Caching Pinecone response in Redis")
                await redis_service.cache_response(message, existing_response)
                
                logger.info("-" * 40)
                logger.info("RESPONSE SOURCE: Pinecone Database")
                logger.info(f"RESPONSE PREVIEW: {existing_response['text'][:100]}...")
                logger.info(f"RESPONSE LENGTH: {len(existing_response['text'])} characters")
                logger.info(f"SIMILARITY SCORE: {similar_responses[0]['score']:.3f}")
                logger.info(f"TOTAL TIME: {time.time() - process_start:.1f}s")
                logger.info("=" * 80)
                return existing_response
            
            pinecone_logger.info("INFO: No sufficiently similar response found in Pinecone (threshold: 0.85)")
            
            # 3. Get X AI response
            xai_logger.info("Step 3: Requesting new response from X AI...")
            xai_start = time.time()
            x_ai_response = await self.generate_content(message)
            xai_end = time.time()
            xai_logger.info(f"SUCCESS: Got response from X AI in {xai_end - xai_start:.1f}s")
            
            # Store in Pinecone for future use
            store_start = time.time()
            embedding = self.get_embedding(message)
            
            metadata = {
                "type": "conversation",
                "message": message,
                "response": x_ai_response.get("text", ""),
                "timestamp": str(datetime.datetime.now())
            }
            
            vector_id = f"conv_{hashlib.md5((message).encode()).hexdigest()}"
            pinecone_logger.info(f"INFO: Storing X AI response in Pinecone [id: {vector_id}]")
            
            try:
                self.index.upsert(
                    vectors=[(vector_id, embedding, metadata)]
                )
                store_end = time.time()
                pinecone_logger.info(f"SUCCESS: Response stored in Pinecone ({store_end - store_start:.1f}s)")
            except Exception as e:
                pinecone_logger.error(f"ERROR: Failed to store in Pinecone: {str(e)}")
                raise
            
            # Cache in Redis
            redis_logger.info("INFO: Caching X AI response in Redis")
            await redis_service.cache_response(message, x_ai_response)
            
            logger.info("-" * 40)
            logger.info("RESPONSE SOURCE: X AI API")
            logger.info(f"RESPONSE PREVIEW: {x_ai_response['text'][:100]}...")
            logger.info(f"RESPONSE LENGTH: {len(x_ai_response['text'])} characters")
            logger.info(f"GENERATION TIME: {xai_end - xai_start:.1f}s")
            logger.info(f"TOTAL TIME: {time.time() - process_start:.1f}s")
            logger.info("=" * 80)

            return x_ai_response

        except Exception as e:
            logger.error("ERROR PROCESSING MESSAGE:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("=" * 80)
            raise

    async def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar content in Pinecone using proper embeddings"""
        search_start = time.time()
        try:
            # Get embedding for query
            pinecone_logger.info(f"Generating embedding for query: '{query}'")
            query_vector = self.get_embedding(query)
            
            # Search in Pinecone
            search_start_time = time.time()
            results = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )
            search_end = time.time()
            pinecone_logger.info(f"✅ Search completed in {search_end - search_start_time:.2f} seconds")
            
            # Convert results to JSON-serializable format
            results_dict = {
                "matches": [{
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": match.metadata
                } for match in results.matches]
            }
            pinecone_logger.debug(f"Search results: {json.dumps(results_dict, indent=2)}")

            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "score": float(match.score),
                    "message": match.metadata.get("message"),
                    "response": match.metadata.get("response"),
                    "timestamp": match.metadata.get("timestamp")
                })

            total_time = time.time() - search_start
            logger.info(f"✨ Total search time: {total_time:.2f} seconds")
            logger.info(f"Found {len(formatted_results)} matches")
            logger.info("=" * 50)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error searching in Pinecone: {str(e)}", exc_info=True)
            raise

    async def add_document(self, title: str, content: str) -> str:
        """Add a document to the vector store"""
        try:
            # Check cache for existing embedding
            cached_embedding = await redis_service.get_cached_embedding(content)
            if cached_embedding:
                doc_embedding = cached_embedding
                logger.info("Using cached document embedding")
            else:
                doc_embedding = await self.get_x_ai_embeddings(content)
                # Cache the embedding
                await redis_service.cache_embedding(content, doc_embedding)
            
            # Store in Pinecone with metadata
            metadata = {
                "type": "document",
                "title": title,
                "content": content
            }
            
            vector_id = f"doc_{hash(title + content)}"
            self.pc.upsert(
                vectors=[(vector_id, doc_embedding, metadata)]
            )
            
            return vector_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise

    async def get_x_ai_embeddings(self, text: str) -> List[float]:
        """Get embeddings from X AI API"""
        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(
                    'https://api.x.ai/v1/embeddings',
                    json={
                        "input": text,
                        "model": "grok-2-1212"  # or whatever model is appropriate for embeddings
                    },
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {settings.X_AI_API_KEY}'
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"X AI API error: Status {response.status_code}, Response: {response.text}")
                    raise Exception("Failed to get embeddings from X AI API")

                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                else:
                    raise Exception("No embedding data in response")

        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

    async def generate_content(self, prompt: str) -> Dict[str, Optional[str]]:
        """Generate content using X AI API"""
        try:
            xai_logger.info(f"Sending request to X AI API with prompt: {prompt[:100]}...")

            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(
                    'https://api.x.ai/v1/chat/completions',
                    json={
                        "model": "grok-2-1212",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a gaming expert and enthusiast AI assistant. You have extensive knowledge about:
- Video games across all platforms (PC, consoles, mobile)
- Gaming hardware and peripherals
- Esports and competitive gaming
- Game mechanics and strategies
- Gaming culture and community

Always provide gaming-focused responses with accurate, helpful information. Include gaming terminology and references when appropriate. Be enthusiastic about gaming while maintaining a friendly, approachable tone.

If a question is not related to gaming, politely explain that you can only help with gaming-related topics."""
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "presence_penalty": 0.6,
                        "frequency_penalty": 0.5
                    },
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {settings.X_AI_API_KEY}'
                    }
                )
                
                if response.status_code != 200:
                    xai_logger.error(f"❌ X AI API error: Status {response.status_code}, Response: {response.text}")
                    return {
                        "text": "I apologize, but I encountered an error. Please try again.",
                        "image_url": None
                    }

                data = response.json()
                xai_logger.info("✅ Successfully received response from X AI API")

                result = {
                    "text": "",
                    "image_url": None
                }

                if data.get("choices") and len(data["choices"]) > 0:
                    result["text"] = data["choices"][0]["message"]["content"]
                    
                    # Handle image URLs if present
                    if "http" in result["text"]:
                        import re
                        urls = re.findall(r'(https?://[^\s]+)', result["text"])
                        if urls:
                            result["image_url"] = urls[0]

                    xai_logger.debug(f"Generated response: {result['text'][:200]}...")
                    return result
                else:
                    xai_logger.warning("⚠️ No choices in X AI response")
                    return {
                        "text": "I apologize, but I couldn't generate a proper response. Please try again.",
                        "image_url": None
                    }

        except Exception as e:
            xai_logger.error(f"❌ Error generating content: {str(e)}", exc_info=True)
            return {
                "text": "I apologize, but something went wrong. Please try again.",
                "image_url": None
            }

rag_service = RAGService() 
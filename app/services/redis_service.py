from typing import Any, Optional
import json
import redis.asyncio as redis
from app.core.config import get_settings
import logging
from datetime import timedelta
import hashlib

settings = get_settings()
logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self):
        self.redis_client = None
        self._pool = None

    async def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.redis_client is None:
                self._pool = redis.ConnectionPool(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=0,
                    decode_responses=True
                )
                self.redis_client = redis.Redis(connection_pool=self._pool)
                logger.info("Redis service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            raise

    async def get_redis(self):
        """Get Redis client, initializing if necessary"""
        if self.redis_client is None:
            await self.initialize_redis()
        return self.redis_client

    async def set_cache(self, key: str, value: Any, expire_seconds: int = 60) -> bool:
        """Set a value in cache with expiration (default 1 minute)"""
        try:
            client = await self.get_redis()
            serialized_value = json.dumps(value)
            await client.set(key, serialized_value, ex=expire_seconds)
            logger.info(f"Set cache for key {key} with {expire_seconds} seconds expiration")
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False

    async def get_cache(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            client = await self.get_redis()
            value = await client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None

    async def delete_cache(self, key: str) -> bool:
        """Delete a value from cache"""
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False

    async def push_to_queue(self, queue_name: str, message: Any) -> bool:
        """Push a message to a queue"""
        try:
            serialized_message = json.dumps(message)
            await self.redis_client.lpush(queue_name, serialized_message)
            return True
        except Exception as e:
            logger.error(f"Error pushing to queue: {str(e)}")
            return False

    async def pop_from_queue(self, queue_name: str) -> Optional[Any]:
        """Pop a message from a queue"""
        try:
            message = await self.redis_client.rpop(queue_name)
            return json.loads(message) if message else None
        except Exception as e:
            logger.error(f"Error popping from queue: {str(e)}")
            return None

    async def get_queue_length(self, queue_name: str) -> int:
        """Get the length of a queue"""
        try:
            return await self.redis_client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error getting queue length: {str(e)}")
            return 0

    async def cache_conversation(self, chat_id: str, messages: list, expire_minutes: int = 30):
        """Cache conversation history"""
        key = f"chat_history:{chat_id}"
        await self.set_cache(key, messages, expire_seconds=expire_minutes * 60)

    async def get_cached_conversation(self, chat_id: str) -> Optional[list]:
        """Get cached conversation history"""
        key = f"chat_history:{chat_id}"
        return await self.get_cache(key)

    async def cache_embedding(self, text: str, embedding: list, expire_hours: int = 1):
        """Cache embeddings for frequently accessed content (1 minute)"""
        key = f"embedding:{hash(text)}"
        await self.set_cache(key, embedding, expire_seconds=60)  # 1 minute cache
        logger.info(f"Cached embedding for text with key {key} for 60 seconds")

    async def get_cached_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding"""
        key = f"embedding:{hash(text)}"
        return await self.get_cache(key)

    async def get_cached_response(self, prompt: str):
        """Get cached response for a given prompt"""
        cache_key = f"response:{hashlib.md5(prompt.encode()).hexdigest()}"
        return await self.get_cache(cache_key)

    async def cache_response(self, prompt: str, response: dict):
        """Cache the response for a given prompt"""
        try:
            cache_key = f"response:{hashlib.md5(prompt.encode()).hexdigest()}"
            # Set cache for 1 minute (60 seconds)
            await self.set_cache(cache_key, response, expire_seconds=60)
            logger.info(f"Cached response with key {cache_key} for 60 seconds")
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")

redis_service = RedisService() 
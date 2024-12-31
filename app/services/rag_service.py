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

settings = get_settings()
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        try:
            # Create an instance of Pinecone
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            logger.info("Pinecone client initialized")
            
            # Connect to the index with specific host
            self.index = self.pc.Index(
                "awais-test",
                host="https://awais-test-8unj2wq.svc.aped-4627-b74a.pinecone.io"
            )
            
            # Get index info to verify dimension
            index_info = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index. Index stats: {index_info}")
            
            # Initialize vector store
            self.vector_store = None
            self.initialize_services()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    def initialize_services(self):
        """Initialize Pinecone and LangChain services"""
        try:
            # Initialize vector store with correct dimension
            self.vector_store = PineconeVectorStore(
                self.index,
                self._text_to_sparse_vector,  # Use our sparse vector function
                "text"
            )
            logger.info("RAG service initialized successfully")
            
            # Test vector dimension
            test_vector = self._text_to_sparse_vector("test")
            logger.info(f"Test vector dimension: {len(test_vector)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    async def process_message(self, message: str, response: str):
        """Store message and response in Pinecone"""
        try:
            # Get the response from X AI first
            x_ai_response = await self.generate_content(message)
            
            # Prepare metadata
            metadata = {
                "type": "conversation",
                "message": message,
                "response": x_ai_response.get("text", ""),
                "timestamp": str(datetime.datetime.now()),
                "normalized_message": ' '.join(message.lower().split())  # Store normalized message
            }

            # Create unique ID for the vector
            vector_id = f"conv_{hashlib.md5((message).encode()).hexdigest()}"
            logger.info(f"Storing vector with ID: {vector_id}")

            # Convert message to vector (only use message for matching)
            vector = self._text_to_sparse_vector(message)
            
            logger.info(f"Vector dimension: {len(vector)}")

            # Store in Pinecone
            try:
                upsert_response = self.index.upsert(
                    vectors=[(
                        vector_id,
                        vector,
                        metadata
                    )]
                )
                logger.info(f"Pinecone upsert response: {upsert_response}")
            except Exception as e:
                logger.error(f"Pinecone upsert error: {str(e)}")
                raise
            
            # Verify the vector was stored
            stats = self.index.describe_index_stats()
            logger.info(f"Current index stats: {stats}")

            return x_ai_response

        except Exception as e:
            logger.error(f"Error storing message in Pinecone: {str(e)}")
            raise

    def _text_to_sparse_vector(self, text: str) -> List[float]:
        """Convert text to sparse vector using consistent hashing"""
        try:
            # Normalize text: lowercase, remove extra spaces, basic cleaning
            text = text.lower()
            text = ' '.join(text.split())  # Remove extra spaces
            text = ''.join(c for c in text if c.isalnum() or c.isspace())  # Remove punctuation
            
            # Create a consistent hash for the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            logger.info(f"Normalized text: '{text}'")
            
            # Generate consistent vector
            vector = np.zeros(1024)
            words = text.split()
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Generate vector components
            for word, freq in word_freq.items():
                # Generate a consistent index for each word
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                index = word_hash % 1024
                
                # Set the value using term frequency
                vector[index] = freq / len(words)
            
            # Normalize the vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            logger.info(f"Generated vector for text: '{text}' with hash: {text_hash[:8]}")
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"Error generating vector: {str(e)}")
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

    async def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar content in Pinecone using sparse vectors"""
        try:
            # Normalize query the same way as stored text
            query = ' '.join(query.lower().split())
            logger.info(f"Searching Pinecone for normalized query: '{query}'")
            
            # Convert query to vector
            query_vector = self._text_to_sparse_vector(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=limit,
                include_metadata=True
            )

            # Format results without logging
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "score": float(match.score),
                    "message": match.metadata.get("message"),
                    "response": match.metadata.get("response"),
                    "timestamp": match.metadata.get("timestamp")
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching in Pinecone: {str(e)}")
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
        """
        Generate content using X AI API with caching and RAG
        """
        try:
            # Generate a cache key from the prompt
            cache_key = f"response:{hashlib.md5(prompt.encode()).hexdigest()}"
            
            # Try to get cached response
            cached_response = await redis_service.get_cache(cache_key)
            if cached_response:
                logger.info("Using cached response")
                return cached_response

            # If no cache hit, generate new response
            is_image_request = any(word in prompt.lower() 
                                 for word in ['image', 'picture', 'draw', 'generate'])

            system_prompt = (
                "You are a witty and humorous AI assistant with a great sense of humor. "
                "Always respond in a lighthearted, entertaining way, incorporating jokes, "
                "wordplay, or funny observations when appropriate. Keep responses clever "
                "and amusing while still being helpful."
            )

            logger.info(f"Sending request to X AI API with prompt: {prompt[:100]}...")

            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(
                    'https://api.x.ai/v1/chat/completions',
                    json={
                        "model": "grok-2-1212",
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.8,
                        "presence_penalty": 0.6,
                        "frequency_penalty": 0.5
                    },
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {settings.X_AI_API_KEY}'
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"X AI API error: Status {response.status_code}, Response: {response.text}")
                    return {
                        "text": "I apologize, but I encountered an error. Please try again.",
                        "image_url": None
                    }

                data = response.json()
                logger.info("Successfully received response from X AI API")

                result = {
                    "text": "",
                    "image_url": None
                }

                if data.get("choices") and len(data["choices"]) > 0:
                    result["text"] = data["choices"][0]["message"]["content"]
                    
                    if is_image_request and "http" in result["text"]:
                        import re
                        urls = re.findall(r'(https?://[^\s]+)', result["text"])
                        if urls:
                            result["image_url"] = urls[0]

                    # Cache the response
                    await redis_service.cache_response(prompt, result)

                return result

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}", exc_info=True)
            return {
                "text": "I apologize, but something went wrong. Please try again.",
                "image_url": None
            }

rag_service = RAGService() 
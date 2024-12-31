# AI Assistant Architecture Documentation

## Overview
This project implements an AI assistant using FastAPI, Telegram Bot, and advanced AI capabilities including RAG (Retrieval-Augmented Generation) powered by LangChain, with data persistence using SQLite and Pinecone vector database, and Redis for caching and message queuing.

## System Components

### 1. FastAPI Application
- Main application server handling HTTP requests and WebSocket connections
- Provides RESTful API endpoints for chat and content generation
- Manages application lifecycle and service initialization

### 2. Telegram Bot Service
- Handles user interactions through Telegram
- Processes commands (/start, /ping, /search)
- Maintains conversation history
- Integrates with RAG and X AI services for response generation

### 3. RAG Service (Retrieval-Augmented Generation)
- Powered by LangChain
- Manages document and conversation embeddings
- Handles similarity search and content retrieval
- Integrates with OpenAI for embeddings and completions
- Uses Redis for caching embeddings and search results

### 4. X AI Service
- Handles image generation requests
- Processes text generation using X AI's API
- Manages API communication and response parsing

### 5. Database Layer
- SQLite: Stores structured data
  - Conversations
  - Documents
  - Metadata
- Pinecone: Vector database
  - Stores embeddings for documents and conversations
  - Enables semantic search capabilities
  - Manages vector similarity operations
- Redis: Caching and Message Queue
  - Caches frequently accessed embeddings
  - Stores temporary conversation history
  - Manages message queues for async processing
  - Improves response times for repeated queries

## Data Flow

### 1. Message Processing Flow
```
User Message → Telegram Bot → Redis Cache Check → RAG Service → Response
   ↓                            ↓                    ↓
   ↓                        1. Check cache       1. Generate embeddings
   ↓                        2. Store result      2. Search similar content
   ↓                                            3. Generate response
   ↓
   → X AI Service (if image requested)
```

### 2. Document Storage Flow
```
Document → RAG Service → 1. Generate/Cache embeddings
                         2. Store in Pinecone
                         3. Store metadata in SQLite
                         4. Cache in Redis
```

### 3. Search Flow
```
Search Query → Redis Cache → If miss → 1. Generate query embedding
                                      2. Search Pinecone
                                      3. Retrieve metadata
                                      4. Cache results
                                      5. Format response
```

## Database Schema

### SQLite Tables

1. Conversations
```sql
- id (Primary Key)
- chat_id (Index)
- user_message
- bot_response
- created_at
- embedding_id (Pinecone reference)
```

2. Documents
```sql
- id (Primary Key)
- title
- content
- embedding_id (Pinecone reference)
- created_at
- updated_at
```

### Pinecone Vectors

1. Structure
```json
{
  "id": "vector_id",
  "values": [embedding_array],
  "metadata": {
    "type": "conversation|document",
    "content": "text content",
    "title": "document title (for documents)",
    "message": "user message (for conversations)",
    "response": "bot response (for conversations)"
  }
}
```

## Redis Schema

### 1. Caching Structure
```
# Embeddings Cache
embedding:{hash(text)} → [embedding_vector]

# Conversation History Cache
chat_history:{chat_id} → [
    {
        "user": "message",
        "assistant": "response"
    }
]

# Search Results Cache
search:{hash(query)} → [
    {
        "score": float,
        "metadata": {
            "type": "conversation|document",
            "content": "text"
        }
    }
]
```

### 2. Message Queue Structure
```
# Processing Queue
processing_queue → [
    {
        "chat_id": "id",
        "message": "text",
        "timestamp": "iso_date"
    }
]

# Response Queue
response_queue → [
    {
        "chat_id": "id",
        "response": "text",
        "timestamp": "iso_date"
    }
]
```

## Environment Configuration

Required environment variables:
```
TELEGRAM_TOKEN=telegram_bot_token
X_AI_API_KEY=x_ai_api_key
OPENAI_API_KEY=openai_api_key
PINECONE_API_KEY=pinecone_api_key
PINECONE_ENV=pinecone_environment
DATABASE_URL=sqlite:///./app.db
PORT=8000

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=optional_password
REDIS_URL=redis://localhost:6379/0
```

## Security Considerations

1. API Security
- All API keys stored in environment variables
- CORS middleware configured
- Rate limiting implemented

2. Data Security
- SQLite database file permissions
- Secure HTTPS connections
- Input validation and sanitization

3. Error Handling
- Comprehensive error logging
- Graceful error responses
- Service recovery mechanisms

## Scaling Considerations

1. Database
- SQLite can be replaced with PostgreSQL for higher load
- Pinecone handles vector scaling automatically

2. Application
- FastAPI supports async operations
- Multiple worker processes possible
- Horizontal scaling supported

3. Memory Management
- Chat history limited to last 5 messages per user
- Periodic cleanup of old conversations
- Efficient vector storage in Pinecone 
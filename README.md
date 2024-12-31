# AI Service with X AI, Redis, and Pinecone

A FastAPI-based AI service that uses X AI for content generation, Redis for caching, and Pinecone for semantic search and retrieval.

## Features

- Content generation using X AI API
- Fast in-memory caching with Redis (1-minute TTL)
- Semantic search using Pinecone vector database
- SQLite for conversation persistence
- Automatic response caching and similarity matching

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd AI_Service
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```env
X_AI_API_KEY=your_x_ai_api_key
PINECONE_API_KEY=your_pinecone_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
```

5. Start Redis server (make sure Redis is installed)

6. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/chat/generate`: Generate responses with caching and similarity matching
- `GET /api/chat/test-pinecone`: Test Pinecone operations
- `POST /api/chat/train`: Train vector database with sample data

## Architecture

- FastAPI for API endpoints
- Redis for fast in-memory caching (1-minute TTL)
- Pinecone for semantic search and vector storage
- SQLite for conversation persistence
- X AI API for content generation

## Caching Strategy

1. Redis Cache (1-minute TTL)
2. Pinecone Similarity Search
3. X AI API Generation

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

## License

MIT License 
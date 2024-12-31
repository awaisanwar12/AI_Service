import asyncio
from app.services.rag_service import rag_service
import logging

logger = logging.getLogger(__name__)

async def train_vector_database():
    """
    Train the vector database with sample data
    """
    sample_data = [
        {
            "question": "What is FastAPI?",
            "answer": "FastAPI is a modern, fast web framework for building APIs with Python. It's like Django's cool, fast cousin who does CrossFit! 🏃‍♂️"
        },
        # Add more sample QA pairs
    ]

    try:
        for data in sample_data:
            await rag_service.process_message(
                data["question"],
                data["answer"]
            )
            logger.info(f"Processed: {data['question']}")

    except Exception as e:
        logger.error(f"Error training vector database: {str(e)}")

if __name__ == "__main__":
    asyncio.run(train_vector_database()) 
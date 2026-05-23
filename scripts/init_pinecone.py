"""
Initialize Pinecone index for the Research Protocol Assistant.

Creates the serverless index with proper configuration.
Run this script once before starting the application:
    python -m scripts.init_pinecone
"""

import logging
import sys
import time

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

# Add project root to path
sys.path.insert(0, ".")

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def init_pinecone():
    """Create the Pinecone index if it doesn't exist."""
    settings = get_settings()

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_name = settings.pinecone_index_name
    dimension = settings.embedding_dimension

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        logger.info("Index '%s' already exists. Checking configuration...", index_name)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info("Index stats: %s", stats)
        logger.info("Namespaces: %s", list(stats.get("namespaces", {}).keys()))
        return

    logger.info(
        "Creating index '%s' with dimension=%d, metric=cosine...",
        index_name,
        dimension,
    )

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Wait for index to be ready
    logger.info("Waiting for index to be ready...")
    while True:
        try:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            logger.info("Index is ready! Stats: %s", stats)
            break
        except Exception:
            logger.info("Index not ready yet, waiting...")
            time.sleep(2)

    logger.info("✅ Pinecone index '%s' created successfully!", index_name)
    logger.info("Namespaces will be auto-created on first upsert:")
    logger.info("  - protocols: Protocol document chunks")
    logger.info("  - memories: Episodic/semantic memories")
    logger.info("  - safety: Safety data and chemical hazards")
    logger.info("  - knowledge: General biology knowledge base")


if __name__ == "__main__":
    init_pinecone()

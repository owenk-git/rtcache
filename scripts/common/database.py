"""
Common database connection utilities.

Extracted from duplicated code across multiple scripts.
"""

import logging
from pymongo import MongoClient
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


class DatabaseConnections:
    """Simple database connection manager."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._mongo_client = None
        self._qdrant_client = None
        self._mongo_db = None
        
    @property
    def mongo_client(self):
        """Lazy-loaded MongoDB client."""
        if self._mongo_client is None:
            self._mongo_client = MongoClient(self.config.database.mongo_url)
            self.logger.info(f"Connected to MongoDB: {self.config.database.mongo_url}")
        return self._mongo_client
    
    @property
    def mongo_db(self):
        """Lazy-loaded MongoDB database."""
        if self._mongo_db is None:
            self._mongo_db = self.mongo_client[self.config.database.mongo_db_name]
        return self._mongo_db
    
    def get_mongo_collection(self, collection_name: str):
        """Get MongoDB collection."""
        return self.mongo_db[collection_name]
    
    @property
    def qdrant_client(self):
        """Lazy-loaded Qdrant client."""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=self.config.database.qdrant_host,
                port=self.config.database.qdrant_port,
                timeout=300
            )
            self.logger.info(f"Connected to Qdrant: {self.config.database.qdrant_host}:{self.config.database.qdrant_port}")
        return self._qdrant_client
    
    def get_qdrant_store(self, collection_name: str):
        """Get Qdrant vector store for llama-index."""
        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            content_key="logical_id"
        )
    
    def close(self):
        """Close all connections."""
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            
        if self._qdrant_client:
            self._qdrant_client.close()
            self._qdrant_client = None
            
        self.logger.info("Database connections closed")


def get_database_connections(config):
    """Simple factory function for database connections."""
    return DatabaseConnections(config)
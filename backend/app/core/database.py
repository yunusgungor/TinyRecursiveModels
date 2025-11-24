"""Database connection and query optimization utilities"""

import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncpg
from asyncpg.pool import Pool

from app.core.config import settings


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager with connection pooling and query optimization
    """
    
    def __init__(self):
        """Initialize database manager"""
        self.pool: Optional[Pool] = None
        self._connection_params = {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
            "database": settings.POSTGRES_DB,
        }
    
    async def connect(self) -> None:
        """
        Create database connection pool
        
        Raises:
            Exception: If connection fails
        """
        try:
            self.pool = await asyncpg.create_pool(
                **self._connection_params,
                min_size=settings.DB_POOL_MIN_SIZE,
                max_size=settings.DB_POOL_MAX_SIZE,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
            )
            
            logger.info("Database connection pool created successfully")
            
            # Create indexes for performance
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def _create_indexes(self) -> None:
        """
        Create database indexes for query optimization
        
        This improves query performance for common access patterns
        """
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                # Create indexes for user profiles
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_profiles_age 
                    ON user_profiles(age)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_profiles_budget 
                    ON user_profiles(budget)
                """)
                
                # Create indexes for gift items
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gift_items_category 
                    ON gift_items(category)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gift_items_price 
                    ON gift_items(price)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gift_items_rating 
                    ON gift_items(rating DESC)
                """)
                
                # Create composite index for common queries
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gift_items_category_price 
                    ON gift_items(category, price)
                """)
                
                # Create index for recommendations
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recommendations_user_id 
                    ON recommendations(user_id)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recommendations_created_at 
                    ON recommendations(created_at DESC)
                """)
                
                logger.info("Database indexes created successfully")
                
        except Exception as e:
            logger.warning(f"Failed to create indexes: {str(e)}")
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool
        
        Yields:
            Database connection
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def execute_query(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a query with timeout
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch_one(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch one row from query
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Row as dictionary or None
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None
    
    async def fetch_all(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows from query
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of rows as dictionaries
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]
    
    async def execute_batch(
        self,
        query: str,
        args_list: List[tuple],
        timeout: Optional[float] = None
    ) -> None:
        """
        Execute batch insert/update
        
        Args:
            query: SQL query
            args_list: List of parameter tuples
            timeout: Query timeout in seconds
        """
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args_list, timeout=timeout)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics
        
        Returns:
            Pool statistics dictionary
        """
        if not self.pool:
            return {"error": "Pool not initialized"}
        
        return {
            "size": self.pool.get_size(),
            "free_size": self.pool.get_idle_size(),
            "max_size": self.pool.get_max_size(),
            "min_size": self.pool.get_min_size(),
        }


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.connect()
    
    return _db_manager


async def close_db_manager() -> None:
    """Close database manager"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.disconnect()
        _db_manager = None

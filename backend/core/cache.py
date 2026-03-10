"""
Caching utilities dengan Redis untuk performance optimization.
"""

import json
import hashlib
from functools import wraps
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Try to import redis, fallback to memory cache if not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using memory cache fallback")


class CacheManager:
    """Manager untuk handling cache operations."""
    
    _instance = None
    _redis_client = None
    _memory_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_redis_client(self):
        """Get atau create Redis client."""
        if not REDIS_AVAILABLE:
            return None
        
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")
                self._redis_client = None
        
        return self._redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value dari cache."""
        client = await self.get_redis_client()
        
        if client:
            try:
                value = await client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback ke memory cache
        return self._memory_cache.get(key)
    
    async def set(self, key: str, value: Any, expiry: int = 300):
        """Set value ke cache."""
        client = await self.get_redis_client()
        
        if client:
            try:
                await client.setex(key, expiry, json.dumps(value))
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Fallback ke memory cache
        self._memory_cache[key] = value
        
        # Cleanup memory cache jika terlalu besar (>1000 items)
        if len(self._memory_cache) > 1000:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
    
    async def delete(self, key: str):
        """Delete key dari cache."""
        client = await self.get_redis_client()
        
        if client:
            try:
                await client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Also remove dari memory cache
        self._memory_cache.pop(key, None)
    
    async def clear_pattern(self, pattern: str):
        """Clear cache keys matching pattern."""
        client = await self.get_redis_client()
        
        if client:
            try:
                keys = await client.keys(pattern)
                if keys:
                    await client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis pattern delete failed: {e}")
        
        # Clear dari memory cache juga
        keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
        for k in keys_to_delete:
            del self._memory_cache[k]


# Global cache manager instance
cache_manager = CacheManager()


def generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key dari function arguments."""
    key_data = f"{func_name}:{str(args)}:{str(kwargs)}"
    return hashlib.md5(key_data.encode()).hexdigest()


def cache_result(expiry: int = 300, key_prefix: str = ""):
    """
    Decorator untuk cache function results.
    
    Args:
        expiry: Cache expiry dalam seconds (default: 5 menit)
        key_prefix: Prefix untuk cache key
        
    Example:
        @cache_result(expiry=600)
        async def get_training_methods():
            # Expensive operation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(func.__name__, args[1:], kwargs)}"
            
            # Try get dari cache
            cached = await cache_manager.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store ke cache
            await cache_manager.set(cache_key, result, expiry)
            logger.debug(f"Cache set for {func.__name__}")
            
            return result
        
        # Add cache invalidation helper
        wrapper.invalidate_cache = lambda: cache_manager.delete(cache_key)
        
        return wrapper
    return decorator


def cache_clear_pattern(pattern: str):
    """Clear cache by pattern."""
    import asyncio
    asyncio.create_task(cache_manager.clear_pattern(pattern))

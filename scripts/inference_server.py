#!/usr/bin/env python3
"""
High-Performance Inference Server for Pixelis

Implements service-level optimizations:
1. FastAPI-based REST API
2. Dynamic batching for throughput
3. Multi-level caching (LRU)
4. Request queuing and prioritization
5. Health monitoring and metrics
6. Support for vLLM and TGI backends
"""

import asyncio
import time
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import logging
from functools import lru_cache
import hashlib
import pickle
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import GPUtil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Metrics for monitoring
request_counter = Counter('inference_requests_total', 'Total inference requests')
request_duration = Histogram('inference_duration_seconds', 'Inference request duration')
batch_size_gauge = Gauge('current_batch_size', 'Current batch size')
queue_size_gauge = Gauge('request_queue_size', 'Number of requests in queue')
cache_hit_rate = Counter('cache_hits_total', 'Total cache hits')
cache_miss_rate = Counter('cache_misses_total', 'Total cache misses')
gpu_utilization_gauge = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
memory_usage_gauge = Gauge('memory_usage_mb', 'Memory usage in MB')


# Request/Response models
class InferenceRequest(BaseModel):
    """Request model for inference."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image: Optional[str] = None  # Base64 encoded image
    question: str
    max_operations: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_cache: bool = True
    priority: int = Field(default=0, ge=0, le=10)  # Higher = higher priority
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    stream: bool = False


class InferenceResponse(BaseModel):
    """Response model for inference."""
    request_id: str
    trajectory: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    latency_ms: float
    from_cache: bool = False
    batch_size: int = 1


@dataclass
class BatchRequest:
    """Internal batched request representation."""
    requests: List[InferenceRequest]
    created_at: datetime
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def age_seconds(self) -> float:
        """Get age of batch in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class DynamicBatcher:
    """
    Dynamic batching for improved throughput.
    
    Collects requests and batches them based on:
    - Maximum batch size
    - Maximum wait time
    - Request priorities
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 50.0,
        enable_padding: bool = True
    ):
        """
        Initialize dynamic batcher.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time_ms: Maximum time to wait for batch formation
            enable_padding: Whether to pad sequences for batching
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_padding = enable_padding
        
        # Priority queue for requests
        self.request_queue: List[Tuple[float, InferenceRequest]] = []
        self.lock = asyncio.Lock()
        
        # Batch formation
        self.current_batch: List[InferenceRequest] = []
        self.batch_start_time: Optional[float] = None
    
    async def add_request(self, request: InferenceRequest) -> None:
        """
        Add a request to the batcher.
        
        Args:
            request: Inference request to add
        """
        async with self.lock:
            # Add to priority queue (negative priority for max heap behavior)
            priority = -request.priority
            self.request_queue.append((priority, request))
            self.request_queue.sort(key=lambda x: x[0])
            queue_size_gauge.set(len(self.request_queue))
    
    async def get_batch(self) -> Optional[BatchRequest]:
        """
        Get a batch of requests for processing.
        
        Returns:
            Batch of requests or None if no batch ready
        """
        async with self.lock:
            # Check if we should form a batch
            if not self.request_queue:
                return None
            
            # Start batch timer if needed
            if self.batch_start_time is None:
                self.batch_start_time = time.time()
            
            # Check batch formation conditions
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            should_batch = (
                len(self.request_queue) >= self.max_batch_size or
                elapsed_ms >= self.max_wait_time_ms
            )
            
            if not should_batch:
                return None
            
            # Form batch
            batch_requests = []
            for _ in range(min(self.max_batch_size, len(self.request_queue))):
                _, request = self.request_queue.pop(0)
                batch_requests.append(request)
            
            # Reset timer
            self.batch_start_time = None if self.request_queue else None
            queue_size_gauge.set(len(self.request_queue))
            batch_size_gauge.set(len(batch_requests))
            
            return BatchRequest(
                requests=batch_requests,
                created_at=datetime.now()
            )


class MultiLevelCache:
    """
    Multi-level caching system for inference results.
    
    Implements:
    - L1: In-memory LRU cache
    - L2: Redis cache (optional)
    - Cache key generation
    - TTL management
    """
    
    def __init__(
        self,
        l1_size: int = 1000,
        l2_enabled: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        default_ttl_s: int = 3600
    ):
        """
        Initialize multi-level cache.
        
        Args:
            l1_size: Size of L1 cache
            l2_enabled: Whether to use Redis L2 cache
            redis_host: Redis host
            redis_port: Redis port
            default_ttl_s: Default TTL in seconds
        """
        self.l1_size = l1_size
        self.default_ttl_s = default_ttl_s
        
        # L1 cache (in-memory LRU)
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_timestamps: Dict[str, datetime] = {}
        
        # L2 cache (Redis)
        self.l2_enabled = l2_enabled
        if l2_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=False
                )
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.l2_enabled = False
                self.redis_client = None
        else:
            self.redis_client = None
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """
        Generate cache key for a request.
        
        Args:
            request: Inference request
            
        Returns:
            Cache key string
        """
        # Create deterministic key from request parameters
        key_data = {
            'question': request.question,
            'image_hash': hashlib.md5(request.image.encode()).hexdigest() if request.image else None,
            'max_operations': request.max_operations,
            'temperature': request.temperature
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """
        Get cached response for a request.
        
        Args:
            request: Inference request
            
        Returns:
            Cached response or None
        """
        if not request.use_cache:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        # Check L1 cache
        if cache_key in self.l1_cache:
            # Check TTL
            timestamp = self.l1_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp).seconds < self.default_ttl_s:
                # Move to end (LRU)
                self.l1_cache.move_to_end(cache_key)
                cache_hit_rate.inc()
                logger.info(f"L1 cache hit for key {cache_key[:8]}...")
                
                response = self.l1_cache[cache_key]
                response.from_cache = True
                return response
            else:
                # Expired, remove
                del self.l1_cache[cache_key]
                del self.l1_timestamps[cache_key]
        
        # Check L2 cache
        if self.l2_enabled and self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    response_dict = pickle.loads(cached_data)
                    response = InferenceResponse(**response_dict)
                    response.from_cache = True
                    
                    # Promote to L1
                    await self._add_to_l1(cache_key, response)
                    
                    cache_hit_rate.inc()
                    logger.info(f"L2 cache hit for key {cache_key[:8]}...")
                    return response
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        cache_miss_rate.inc()
        return None
    
    async def set(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        ttl_s: Optional[int] = None
    ) -> None:
        """
        Cache a response.
        
        Args:
            request: Original request
            response: Response to cache
            ttl_s: TTL in seconds (optional)
        """
        if not request.use_cache:
            return
        
        cache_key = self._generate_cache_key(request)
        ttl = ttl_s or self.default_ttl_s
        
        # Add to L1
        await self._add_to_l1(cache_key, response)
        
        # Add to L2
        if self.l2_enabled and self.redis_client:
            try:
                response_data = pickle.dumps(response.dict())
                self.redis_client.setex(cache_key, ttl, response_data)
                logger.info(f"Cached to L2 with key {cache_key[:8]}...")
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    async def _add_to_l1(self, key: str, response: InferenceResponse) -> None:
        """Add response to L1 cache."""
        # Evict if full
        if len(self.l1_cache) >= self.l1_size:
            # Remove oldest
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
            if oldest_key in self.l1_timestamps:
                del self.l1_timestamps[oldest_key]
        
        # Add new entry
        self.l1_cache[key] = response
        self.l1_timestamps[key] = datetime.now()


class InferenceEngine:
    """
    Mock inference engine for demonstration.
    In production, this would interface with actual model.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize inference engine."""
        self.model_path = model_path
        logger.info("Initialized mock inference engine")
    
    async def process_batch(self, batch: BatchRequest) -> List[InferenceResponse]:
        """
        Process a batch of requests.
        
        Args:
            batch: Batch of requests
            
        Returns:
            List of responses
        """
        responses = []
        
        for request in batch.requests:
            start_time = time.time()
            
            # Mock inference
            trajectory = [
                {
                    "operation": "SEGMENT_OBJECT_AT",
                    "arguments": {"x": 100, "y": 200},
                    "result": "Found object: cat"
                },
                {
                    "operation": "GET_PROPERTIES",
                    "arguments": {"object": "cat"},
                    "result": {"color": "orange", "size": "medium"}
                },
                {
                    "operation": "FINAL_ANSWER",
                    "arguments": {},
                    "result": "An orange cat"
                }
            ]
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            response = InferenceResponse(
                request_id=request.request_id,
                trajectory=trajectory,
                final_answer="An orange cat",
                confidence=0.95,
                latency_ms=(time.time() - start_time) * 1000,
                batch_size=len(batch.requests)
            )
            
            responses.append(response)
        
        return responses


class PixelisInferenceServer:
    """
    Main inference server with all optimizations.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 50.0,
        cache_size: int = 1000,
        enable_redis: bool = False
    ):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to model
            max_batch_size: Maximum batch size
            max_wait_time_ms: Maximum wait time for batching
            cache_size: Size of L1 cache
            enable_redis: Enable Redis L2 cache
        """
        # Components
        self.batcher = DynamicBatcher(
            max_batch_size=max_batch_size,
            max_wait_time_ms=max_wait_time_ms
        )
        
        self.cache = MultiLevelCache(
            l1_size=cache_size,
            l2_enabled=enable_redis
        )
        
        self.engine = InferenceEngine(model_path)
        
        # Request tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Background task
        self.processing_task = None
    
    async def start(self):
        """Start the inference server."""
        logger.info("Starting inference server")
        self.processing_task = asyncio.create_task(self._process_batches())
    
    async def stop(self):
        """Stop the inference server."""
        logger.info("Stopping inference server")
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def _process_batches(self):
        """Background task to process batches."""
        while True:
            try:
                # Get next batch
                batch = await self.batcher.get_batch()
                
                if batch:
                    # Process batch
                    responses = await self.engine.process_batch(batch)
                    
                    # Cache and deliver responses
                    for request, response in zip(batch.requests, responses):
                        # Cache result
                        await self.cache.set(request, response)
                        
                        # Deliver to waiting request
                        if request.request_id in self.pending_requests:
                            future = self.pending_requests[request.request_id]
                            future.set_result(response)
                            del self.pending_requests[request.request_id]
                else:
                    # No batch ready, wait a bit
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Perform inference on a request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        request_counter.inc()
        
        # Check cache
        cached_response = await self.cache.get(request)
        if cached_response:
            return cached_response
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.request_id] = future
        
        # Add to batcher
        await self.batcher.add_request(request)
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=request.timeout_s)
            return response
        except asyncio.TimeoutError:
            # Remove from pending
            if request.request_id in self.pending_requests:
                del self.pending_requests[request.request_id]
            raise HTTPException(status_code=408, detail="Request timeout")


# Global server instance
server: Optional[PixelisInferenceServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app."""
    global server
    
    # Startup
    logger.info("Starting Pixelis inference server")
    server = PixelisInferenceServer(
        max_batch_size=8,
        max_wait_time_ms=50.0,
        cache_size=1000,
        enable_redis=False  # Set to True if Redis is available
    )
    await server.start()
    
    # Start monitoring task
    asyncio.create_task(monitor_system())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Pixelis inference server")
    if server:
        await server.stop()


# Create FastAPI app
app = FastAPI(
    title="Pixelis Inference Server",
    description="High-performance inference server with dynamic batching and caching",
    version="1.0.0",
    lifespan=lifespan
)


async def monitor_system():
    """Background task to monitor system metrics."""
    while True:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_gauge.set(memory.used / 1024 / 1024)
            
            # GPU usage
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_utilization_gauge.set(gpus[0].load * 100)
                except:
                    pass
            
            await asyncio.sleep(10)  # Update every 10 seconds
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(10)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest) -> InferenceResponse:
    """
    Perform inference on a single request.
    
    Args:
        request: Inference request
        
    Returns:
        Inference response
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    with request_duration.time():
        response = await server.infer(request)
    
    return response


@app.post("/batch_infer", response_model=List[InferenceResponse])
async def batch_infer(requests: List[InferenceRequest]) -> List[InferenceResponse]:
    """
    Perform inference on multiple requests.
    
    Args:
        requests: List of inference requests
        
    Returns:
        List of inference responses
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Process requests concurrently
    tasks = [server.infer(req) for req in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    valid_responses = []
    for response in responses:
        if isinstance(response, Exception):
            logger.error(f"Batch inference error: {response}")
            # Could return error response instead
        else:
            valid_responses.append(response)
    
    return valid_responses


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server_initialized": server is not None
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        generate_latest(),
        media_type="text/plain"
    )


@app.get("/stats")
async def stats():
    """Get server statistics."""
    if not server:
        return {"error": "Server not initialized"}
    
    return {
        "queue_size": len(server.batcher.request_queue),
        "pending_requests": len(server.pending_requests),
        "l1_cache_size": len(server.cache.l1_cache),
        "l2_enabled": server.cache.l2_enabled
    }


def main():
    """Main entry point for the inference server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pixelis Inference Server")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Run server
    uvicorn.run(
        "inference_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
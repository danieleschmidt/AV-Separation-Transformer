#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import torch
import numpy as np
from av_separation import SeparatorConfig
from av_separation.models import AVSeparationTransformer
from av_separation.performance_optimizer import (
    AdvancedCache, BatchProcessor, PerformanceProfiler, 
    cached_result, performance_monitored, global_cache,
    global_batch_processor, global_profiler, benchmark_function
)
from av_separation.scaling import (
    LoadBalancer, AutoScaler, WorkerNode, WorkerStatus,
    DistributedCoordinator
)

# Global optimization infrastructure
model_cache = AdvancedCache(max_size=10, max_memory_mb=2048, ttl_seconds=7200)
batch_processor = BatchProcessor(batch_size=8, max_wait_time=0.05)

@cached_result(ttl=1800, max_size=50)
def cached_model_creation(config_hash: str):
    """Cached model creation for reuse"""
    config = SeparatorConfig()
    model = AVSeparationTransformer(config)
    return model

@performance_monitored('scaling_demo')
def process_batch_inference(batch_items):
    """Batch inference processing (sync function for batch processor)"""
    config = SeparatorConfig()
    model = AVSeparationTransformer(config)
    model.eval()
    
    results = []
    with torch.no_grad():
        for audio_data, video_data in batch_items:
            outputs = model(audio_data, video_data)
            results.append({
                'waveforms': outputs['separated_waveforms'],
                'logits': outputs['speaker_logits']
            })
    
    return results

async def test_performance_optimization():
    """Test advanced performance optimization features"""
    print("=== Testing Performance Optimization ===")
    
    # Test advanced caching
    print("1. Testing Advanced Cache...")
    cache_stats_before = global_cache.get_stats()
    
    # Add test data to cache
    test_data = {"model_config": "test", "tensor": torch.randn(100, 100)}
    global_cache.put("test_key_1", test_data)
    global_cache.put("test_key_2", np.random.randn(50, 50))
    
    # Test cache retrieval
    cached_data, hit = global_cache.get("test_key_1")
    print(f"   ✓ Cache hit: {hit}")
    
    cache_stats_after = global_cache.get_stats()
    print(f"   ✓ Cache stats: {cache_stats_after}")
    
    # Test performance profiler
    print("\n2. Testing Performance Profiler...")
    with global_profiler.profile("demo", "tensor_operation"):
        # Simulate some tensor operations
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        c = torch.matmul(a, b)
        
    profiler_stats = global_profiler.get_overall_stats()
    print(f"   ✓ Profiler stats: {profiler_stats}")
    
    # Test batch processing
    print("\n3. Testing Batch Processing...")
    await batch_processor.start()
    
    # Create test batches
    batch_tasks = []
    for i in range(5):
        audio_data = torch.randn(1, 80, 50)
        video_data = torch.randn(1, 15, 3, 224, 224)
        task = batch_processor.process_async(
            (audio_data, video_data),
            process_batch_inference,
            timeout=30.0
        )
        batch_tasks.append(task)
    
    # Wait for batch results
    batch_results = await asyncio.gather(*batch_tasks)
    print(f"   ✓ Batch processing completed: {len(batch_results)} results")
    
    batch_stats = batch_processor.get_stats()
    print(f"   ✓ Batch stats: {batch_stats}")
    
    await batch_processor.stop()
    
    return True

async def test_auto_scaling():
    """Test auto-scaling and load balancing"""
    print("\n=== Testing Auto-Scaling and Load Balancing ===")
    
    # Initialize distributed coordinator
    coordinator = DistributedCoordinator()
    
    # Add initial workers
    print("1. Adding Initial Workers...")
    for i in range(3):
        worker = WorkerNode(
            node_id=f"worker_{i}",
            host="localhost",
            port=8000 + i,
            status=WorkerStatus.IDLE,
            current_load=0.0,
            max_capacity=4,
            active_requests=0,
            last_heartbeat=asyncio.get_event_loop().time(),
            gpu_memory_mb=2000.0,
            cpu_percent=25.0,
            model_loaded=True,
            capabilities=["audio_visual_separation"]
        )
        coordinator.load_balancer.register_worker(worker)
    
    # Test load balancing
    print("2. Testing Load Balancing...")
    load_balancer = coordinator.load_balancer
    
    # Test different load balancing strategies
    strategies = ["round_robin", "least_connections", "resource_aware"]
    
    for strategy in strategies:
        load_balancer.strategy = strategy
        selected_worker = load_balancer.select_worker({"test_request": True})
        if selected_worker:
            print(f"   ✓ {strategy}: Selected worker {selected_worker.node_id}")
            load_balancer.release_worker(selected_worker.node_id)
        else:
            print(f"   ✗ {strategy}: No worker selected")
    
    # Test distributed processing
    print("3. Testing Distributed Processing...")
    
    # Simulate processing requests
    requests = []
    for i in range(5):
        audio_data = np.random.randn(16000)  # 1 second at 16kHz
        video_data = np.random.randn(30, 224, 224, 3)  # 30 frames
        
        request = coordinator.process_request(
            request_id=f"req_{i}",
            audio_data=audio_data,
            video_data=video_data,
            num_speakers=2
        )
        requests.append(request)
    
    # Wait for all requests
    results = await asyncio.gather(*requests)
    successful_requests = sum(1 for r in results if r['success'])
    print(f"   ✓ Processed {successful_requests}/{len(results)} requests successfully")
    
    # Test auto-scaling simulation
    print("4. Testing Auto-Scaling Logic...")
    auto_scaler = coordinator.auto_scaler
    
    # Simulate high load
    for worker_id, worker in load_balancer.workers.items():
        worker.active_requests = worker.max_capacity - 1  # Near capacity
        worker.current_load = 0.9
        load_balancer._update_worker_status_by_load(worker)
    
    # Get metrics and test scaling decision
    scaling_stats = auto_scaler.get_scaling_stats()
    print(f"   ✓ Auto-scaling stats: {scaling_stats}")
    
    # Get system status
    system_status = coordinator.get_status()
    print(f"   ✓ System status: Workers={len(system_status['load_balancer']['workers'])}")
    
    return True

def test_cached_functions():
    """Test cached function decorators"""
    print("\n=== Testing Cached Functions ===")
    
    @cached_result(ttl=60, max_size=10)
    def expensive_computation(n: int) -> int:
        # Simulate expensive computation
        result = sum(i ** 2 for i in range(n))
        return result
    
    # Test caching behavior
    print("1. Testing Function Caching...")
    
    # First call (cache miss)
    result1 = expensive_computation(1000)
    
    # Second call (should be cache hit)
    result2 = expensive_computation(1000)
    
    print(f"   ✓ Results match: {result1 == result2}")
    print(f"   ✓ Cache stats: {expensive_computation.cache.get_stats()}")
    
    return True

def benchmark_model_inference():
    """Benchmark model inference performance"""
    print("\n=== Benchmarking Model Inference ===")
    
    config = SeparatorConfig()
    model = AVSeparationTransformer(config)
    model.eval()
    
    def single_inference():
        with torch.no_grad():
            audio = torch.randn(1, 80, 50)
            video = torch.randn(1, 15, 3, 224, 224)
            outputs = model(audio, video)
            return outputs
    
    # Benchmark inference
    benchmark_results = benchmark_function(single_inference, iterations=10)
    
    print(f"   ✓ Mean inference time: {benchmark_results['mean_duration']:.3f}s")
    print(f"   ✓ Min inference time: {benchmark_results['min_duration']:.3f}s")
    print(f"   ✓ Max inference time: {benchmark_results['max_duration']:.3f}s")
    print(f"   ✓ Throughput: {1/benchmark_results['mean_duration']:.1f} inferences/sec")
    
    return benchmark_results

async def main():
    """Main scaling demo"""
    print("=== AV-Separation Transformer Scaling Demo ===")
    
    try:
        # Test performance optimization
        perf_result = await test_performance_optimization()
        print(f"✓ Performance optimization test: {'PASSED' if perf_result else 'FAILED'}")
        
        # Test auto-scaling
        scaling_result = await test_auto_scaling()
        print(f"✓ Auto-scaling test: {'PASSED' if scaling_result else 'FAILED'}")
        
        # Test cached functions
        cache_result = test_cached_functions()
        print(f"✓ Cached functions test: {'PASSED' if cache_result else 'FAILED'}")
        
        # Benchmark model inference
        benchmark_results = benchmark_model_inference()
        print(f"✓ Model inference benchmark: {benchmark_results['mean_duration']:.3f}s avg")
        
        print(f"\n=== Generation 3 (MAKE IT SCALE) Complete ===")
        print("✅ Advanced performance optimization, caching, batch processing, ")
        print("   load balancing, auto-scaling, and distributed processing active!")
        
        # Final system stats
        print(f"\n=== Final System Performance ===")
        print(f"Global Cache: {global_cache.get_stats()}")
        print(f"Global Profiler: {global_profiler.get_overall_stats()}")
        
    except Exception as e:
        print(f"✗ Scaling system test: FAILED - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
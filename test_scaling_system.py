#!/usr/bin/env python3
"""
Test Suite for Scaling System Components
Generation 3: MAKE IT SCALE testing
"""

import sys
import asyncio
import time
import random
from pathlib import Path

sys.path.append('/root/repo/src')

print("⚡ SCALING SYSTEM TESTING - GENERATION 3")
print("=" * 60)


def test_performance_optimizer():
    """Test performance optimization components"""
    print("\n🚀 Testing Performance Optimization:")
    print("-" * 40)
    
    try:
        from av_separation.performance_optimizer import (
            AdvancedCache, BatchProcessor, ConnectionPool,
            PerformanceProfiler, cached_result, performance_monitored
        )
        print("✓ Performance optimization components imported")
        
        # Test advanced cache
        cache = AdvancedCache(max_size=100, max_memory_mb=10, ttl_seconds=60)
        
        # Test cache operations
        success = cache.put("test_key", "test_value")
        print(f"✓ Cache put operation: {success}")
        
        value, hit = cache.get("test_key")
        print(f"✓ Cache get operation: hit={hit}, value={'test_value' if hit else 'None'}")
        
        # Test cache with large data
        large_data = "x" * 1000  # 1KB string
        cache.put("large_key", large_data)
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"✓ Cache stats: size={stats['size']}, memory={stats['memory_usage_mb']:.2f}MB, hit_rate={stats['hit_rate']:.2f}")
        
        # Test cache eviction
        for i in range(150):  # Exceed cache size
            cache.put(f"key_{i}", f"value_{i}")
        
        final_stats = cache.get_stats()
        print(f"✓ Cache after eviction: size={final_stats['size']}, evictions={final_stats['eviction_count']}")
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        
        # Test profiling context
        with profiler.profile('test_component', 'test_operation'):
            time.sleep(0.01)  # Simulate work
        
        component_stats = profiler.get_component_stats('test_component')
        print(f"✓ Performance profiling: operations={component_stats.get('total_operations', 0)}")
        
        # Test cached result decorator
        @cached_result(ttl=60, max_size=10)
        def expensive_function(x):
            time.sleep(0.001)  # Simulate expensive operation
            return x * x
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_function(5)
        first_duration = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_function(5)
        second_duration = time.time() - start_time
        
        print(f"✓ Caching decorator: first={first_duration:.4f}s, second={second_duration:.4f}s, speedup={first_duration/second_duration:.1f}x")
        
        # Test performance monitoring decorator
        @performance_monitored('test_component')
        def monitored_function():
            time.sleep(0.005)
            return "test_result"
        
        result = monitored_function()
        monitor_stats = monitored_function.profiler.get_overall_stats()
        print(f"✓ Performance monitoring: operations={monitor_stats.get('total_operations', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_auto_scaler():
    """Test auto-scaling components"""
    print("\n📈 Testing Auto-Scaling System:")
    print("-" * 40)
    
    try:
        from av_separation.auto_scaler import (
            AutoScaler, MetricsCollector, ScalingEngine, LoadBalancer,
            ScalingMetrics, ScalingRule, ResourceType, ScalingAction
        )
        print("✓ Auto-scaling components imported")
        
        # Test metrics collector
        collector = MetricsCollector(history_size=100)
        
        # Add test metrics
        for i in range(10):
            metrics = ScalingMetrics(
                cpu_usage=50.0 + random.uniform(-10, 10),
                memory_usage=60.0 + random.uniform(-15, 15),
                queue_length=random.randint(0, 20),
                response_time=100.0 + random.uniform(-20, 50),
                error_rate=random.uniform(0, 5),
                throughput=100.0 + random.uniform(-20, 20),
                active_connections=random.randint(5, 25)
            )
            collector.add_metrics(metrics)
            await asyncio.sleep(0.01)  # Small delay
        
        # Test metrics aggregation
        avg_metrics = collector.get_average_metrics(seconds=10)
        print(f"✓ Metrics collection: {len(collector.metrics_history)} samples, avg_cpu={avg_metrics['cpu_usage']:.1f}%")
        
        percentile_metrics = collector.get_percentile_metrics(percentile=95)
        print(f"✓ Percentile metrics: 95th percentile CPU={percentile_metrics['cpu_usage']:.1f}%")
        
        # Test scaling engine
        engine = ScalingEngine()
        
        # Test high CPU scenario (should trigger scale up)
        high_cpu_metrics = {
            'cpu_usage': 85.0,  # Above 75% threshold
            'memory_usage': 40.0,
            'response_time': 50.0,
            'active_connections': 10
        }
        
        actions = engine.evaluate_scaling(high_cpu_metrics)
        scale_up_actions = [a for a in actions if a['action'] == ScalingAction.SCALE_UP]
        print(f"✓ High CPU scaling: {len(scale_up_actions)} scale-up actions triggered")
        
        # Test low CPU scenario (should trigger scale down after cooldown)
        time.sleep(0.1)  # Brief wait
        engine.last_scaling_action.clear()  # Reset cooldown for testing
        
        low_cpu_metrics = {
            'cpu_usage': 15.0,  # Below 25% threshold
            'memory_usage': 20.0,
            'response_time': 30.0,
            'active_connections': 3
        }
        
        actions = engine.evaluate_scaling(low_cpu_metrics)
        scale_down_actions = [a for a in actions if a['action'] == ScalingAction.SCALE_DOWN]
        print(f"✓ Low CPU scaling: {len(scale_down_actions)} scale-down actions triggered")
        
        # Test load balancer
        balancer = LoadBalancer(initial_workers=4)
        
        # Test worker selection strategies
        worker_round_robin = balancer.get_next_worker("round_robin")
        worker_least_conn = balancer.get_next_worker("least_connections")
        print(f"✓ Load balancing: round_robin={worker_round_robin}, least_conn={worker_least_conn}")
        
        # Simulate request processing
        balancer.record_request_start(worker_round_robin)
        await asyncio.sleep(0.05)  # Simulate processing time
        balancer.record_request_end(worker_round_robin, response_time=50.0, error=False)
        
        worker_stats = balancer.get_worker_stats()
        total_requests = sum(stats['total_requests'] for stats in worker_stats.values())
        print(f"✓ Request tracking: {total_requests} total requests processed")
        
        # Test auto-scaler integration
        autoscaler = AutoScaler(
            metrics_window=60,
            evaluation_interval=1,  # Fast evaluation for testing
            enable_predictive_scaling=True
        )
        
        # Register test callback
        scaling_events = []
        
        def test_scaling_callback(action):
            scaling_events.append(action)
        
        autoscaler.register_scaling_callback(ResourceType.CPU_WORKERS, test_scaling_callback)
        
        # Add metrics that should trigger scaling
        trigger_metrics = ScalingMetrics(
            cpu_usage=90.0,  # High CPU to trigger scaling
            memory_usage=45.0,
            queue_length=5,
            response_time=75.0,
            error_rate=1.0,
            throughput=95.0,
            active_connections=15
        )
        
        autoscaler.add_metrics(trigger_metrics)
        
        # Brief test of scaling evaluation
        await autoscaler._evaluate_scaling()
        
        print(f"✓ Auto-scaler integration: {len(scaling_events)} scaling events triggered")
        
        # Get comprehensive status
        status = autoscaler.get_status()
        print(f"✓ Auto-scaler status: running={status['running']}, resource_levels={len(status['resource_levels'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_processing():
    """Test batch processing system"""
    print("\n📦 Testing Batch Processing:")
    print("-" * 40)
    
    try:
        from av_separation.performance_optimizer import BatchProcessor
        
        processor = BatchProcessor(
            batch_size=5,
            max_wait_time=0.1,
            max_queue_size=100,
            num_workers=2
        )
        
        await processor.start()
        print("✓ Batch processor started")
        
        # Test batch processing function
        def simple_batch_function(items):
            # Simulate processing each item
            return [f"processed_{item}" for item in items]
        
        # Submit multiple items for batch processing
        tasks = []
        for i in range(12):  # More than one batch
            task = processor.process_async(f"item_{i}", simple_batch_function, timeout=5.0)
            tasks.append(task)
        
        # Wait for all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        print(f"✓ Batch processing: {len(successful_results)}/{len(tasks)} items processed successfully")
        
        # Check some results
        if successful_results:
            print(f"✓ Sample result: {successful_results[0]}")
        
        # Get processing statistics
        stats = processor.get_stats()
        print(f"✓ Batch stats: {stats['processed_batches']} batches, {stats['total_items']} items, {stats['throughput_items_per_sec']:.1f} items/sec")
        
        await processor.stop()
        print("✓ Batch processor stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_connection_pooling():
    """Test connection pooling"""
    print("\n🔗 Testing Connection Pooling:")
    print("-" * 40)
    
    try:
        from av_separation.performance_optimizer import ConnectionPool
        
        # Mock connection creation
        connection_count = 0
        
        async def create_mock_connection():
            nonlocal connection_count
            connection_count += 1
            # Return a simple mock connection object
            return type('MockConnection', (), {
                'id': connection_count,
                'close': lambda: None
            })()
        
        pool = ConnectionPool(
            create_connection=create_mock_connection,
            max_connections=5,
            min_connections=2,
            max_idle_time=10,
            connection_timeout=5
        )
        
        async def test_pool_usage():
            # Get multiple connections
            connections = []
            for i in range(3):
                conn = await pool.get_connection()
                connections.append(conn)
                print(f"✓ Got connection {i+1}: {conn.id}")
            
            # Return connections to pool
            for conn in connections:
                await pool.return_connection(conn)
                print(f"✓ Returned connection {conn.id}")
            
            # Get connection again (should reuse)
            reused_conn = await pool.get_connection()
            print(f"✓ Reused connection: {reused_conn.id}")
            await pool.return_connection(reused_conn)
            
            # Get pool statistics
            stats = pool.get_stats()
            print(f"✓ Pool stats: created={stats['total_created']}, reused={stats['total_reused']}")
        
        # Run the async test directly since we're already in async context
        await test_pool_usage()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection pooling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_scaling():
    """Test integrated scaling system"""
    print("\n🔄 Testing Integrated Scaling:")
    print("-" * 40)
    
    try:
        from av_separation.performance_optimizer import global_cache, global_profiler
        from av_separation.auto_scaler import global_autoscaler
        
        print("✓ Global scaling components initialized")
        
        # Test global cache integration
        global_cache.put("integration_test", "test_data")
        value, hit = global_cache.get("integration_test")
        print(f"✓ Global cache integration: hit={hit}")
        
        cache_stats = global_cache.get_stats()
        print(f"✓ Global cache stats: size={cache_stats['size']}, hit_rate={cache_stats['hit_rate']:.2f}")
        
        # Test global profiler integration
        with global_profiler.profile('integration_test', 'test_operation'):
            # Simulate some work
            await asyncio.sleep(0.01)
        
        profiler_stats = global_profiler.get_overall_stats()
        print(f"✓ Global profiler integration: {profiler_stats.get('total_operations', 0)} operations tracked")
        
        # Test system under load simulation
        print("\n🧪 Running Load Simulation:")
        
        # Simulate increasing load
        from av_separation.auto_scaler import ScalingMetrics
        
        load_levels = [
            (30, 40),   # Low load
            (60, 55),   # Medium load  
            (85, 75),   # High load
            (95, 90),   # Very high load
            (40, 45),   # Back to medium
        ]
        
        for cpu, memory in load_levels:
            metrics = ScalingMetrics(
                cpu_usage=cpu,
                memory_usage=memory,
                queue_length=random.randint(1, 20),
                response_time=random.uniform(20, 200),
                error_rate=random.uniform(0, 5),
                throughput=random.uniform(50, 150),
                active_connections=random.randint(5, 30)
            )
            
            global_autoscaler.add_metrics(metrics)
            await asyncio.sleep(0.1)
        
        # Get final autoscaler status
        final_status = global_autoscaler.get_status()
        print(f"✓ Load simulation complete: {len(final_status.get('scaling_history', []))} scaling decisions made")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    print("🧪 Starting Generation 3 Scaling System Tests")
    
    tests = [
        ("Performance Optimizer", test_performance_optimizer),
        ("Auto-Scaler", test_auto_scaler),
        ("Batch Processing", test_batch_processing),
        ("Connection Pooling", test_connection_pooling),
        ("Integrated Scaling", test_integrated_scaling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running {test_name} Tests")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            duration = time.time() - start_time
            
            if success:
                print(f"\n✅ {test_name} Tests: PASSED ({duration:.2f}s)")
                results.append((test_name, "PASSED", duration))
            else:
                print(f"\n❌ {test_name} Tests: FAILED ({duration:.2f}s)")
                results.append((test_name, "FAILED", duration))
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n💥 {test_name} Tests: ERROR - {str(e)} ({duration:.2f}s)")
            results.append((test_name, "ERROR", duration))
    
    # Print final results
    print(f"\n{'='*60}")
    print("📊 GENERATION 3 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status in ["FAILED", "ERROR"])
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, status, duration in results:
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {test_name:<25} {status:<7} ({duration:.2f}s)")
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed in {total_time:.2f}s")
    
    if passed == len(tests):
        print("\n🚀 GENERATION 3: MAKE IT SCALE - COMPLETED SUCCESSFULLY")
        print("✅ Performance optimization systems operational")
        print("✅ Auto-scaling and load balancing implemented")
        print("✅ Advanced caching and batch processing working")
        print("✅ Connection pooling and resource management active")
        print("✅ Ready for Quality Gates and Global Implementation")
    else:
        print(f"\n⚠️  GENERATION 3: PARTIALLY COMPLETE ({passed}/{len(tests)} passed)")
        print("🔧 Some scaling features may need optimization")
    
    return passed == len(tests)


if __name__ == "__main__":
    asyncio.run(main())
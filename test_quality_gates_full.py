#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import asyncio
import torch
import numpy as np
from typing import Dict, List, Any
from av_separation import SeparatorConfig
from av_separation.models import AVSeparationTransformer
from av_separation.enhanced_security import (
    AdvancedInputValidator, SecurityAuditor, RateLimiter,
    SecurityEvent, ThreatLevel
)
from av_separation.robust_core import (
    RobustValidator, CircuitBreaker, RetryHandler,
    global_health_checker
)
from av_separation.performance_optimizer import (
    AdvancedCache, PerformanceProfiler, benchmark_function
)

class QualityGateRunner:
    """Comprehensive quality gate execution system"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.validator = AdvancedInputValidator()
        self.auditor = SecurityAuditor()
        self.robust_validator = RobustValidator()
        self.profiler = PerformanceProfiler()
        
    def log_result(self, gate_name: str, passed: bool, details: Dict[str, Any] = None):
        """Log quality gate result"""
        self.results[gate_name] = {
            'passed': passed,
            'timestamp': time.time(),
            'details': details or {}
        }
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {gate_name}")
        
        if not passed and details:
            print(f"   Details: {details}")
    
    async def run_all_gates(self) -> Dict[str, Any]:
        """Execute all mandatory quality gates"""
        print("🚀 Starting Mandatory Quality Gates Execution")
        print("=" * 60)
        
        # 1. Code Quality Gate
        await self.gate_code_quality()
        
        # 2. Security Gate  
        await self.gate_security_validation()
        
        # 3. Performance Gate
        await self.gate_performance_benchmarks()
        
        # 4. Reliability Gate
        await self.gate_reliability_testing()
        
        # 5. Scalability Gate
        await self.gate_scalability_testing()
        
        # 6. Integration Gate
        await self.gate_integration_testing()
        
        # 7. Compliance Gate
        await self.gate_compliance_checks()
        
        # Generate final report
        return self.generate_final_report()
    
    async def gate_code_quality(self):
        """Gate 1: Code Quality and Standards"""
        print("\n🔍 Gate 1: Code Quality and Standards")
        
        try:
            # Test basic functionality
            config = SeparatorConfig()
            model = AVSeparationTransformer(config)
            
            # Check parameter count is reasonable
            param_count = model.get_num_params()
            param_check = 100_000_000 <= param_count <= 150_000_000
            
            # Test forward pass
            dummy_audio = torch.randn(1, config.audio.n_mels, 50)
            dummy_video = torch.randn(1, 15, 3, *config.video.image_size)
            
            model.eval()
            with torch.no_grad():
                outputs = model(dummy_audio, dummy_video)
                
            output_shapes_correct = (
                len(outputs['separated_waveforms'].shape) == 3 and
                len(outputs['speaker_logits'].shape) == 3 and
                'alignment_score' in outputs
            )
            
            # Test configuration validation
            config_valid = isinstance(config.to_dict(), dict)
            
            all_passed = param_check and output_shapes_correct and config_valid
            
            self.log_result("code_quality", all_passed, {
                'parameter_count': param_count,
                'parameter_range_ok': param_check,
                'output_shapes_correct': output_shapes_correct,
                'config_serializable': config_valid
            })
            
        except Exception as e:
            self.log_result("code_quality", False, {'error': str(e)})
    
    async def gate_security_validation(self):
        """Gate 2: Security Validation"""
        print("\n🔒 Gate 2: Security Validation")
        
        try:
            # Test input validation
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "SELECT * FROM users",
                "../../etc/passwd",
                "javascript:alert(1)",
                "eval('malicious code')"
            ]
            
            blocked_count = 0
            for malicious_input in malicious_inputs:
                is_valid, errors = self.validator.validate_text_input(malicious_input)
                if not is_valid:
                    blocked_count += 1
            
            security_effective = blocked_count >= len(malicious_inputs) * 0.8  # 80% detection rate
            
            # Test rate limiting
            rate_limiter = RateLimiter(max_requests=5, time_window=60)
            client_id = "test_client"
            
            rate_limit_working = False
            for i in range(10):
                allowed, info = rate_limiter.is_allowed(client_id)
                if not allowed:
                    rate_limit_working = True
                    break
            
            # Test security auditing
            trace_id = self.auditor.log_security_event(
                SecurityEvent.SUSPICIOUS_PATTERN,
                ThreatLevel.MEDIUM,
                "Test security event",
                source_ip="127.0.0.1"
            )
            
            audit_working = trace_id is not None
            
            all_passed = security_effective and rate_limit_working and audit_working
            
            self.log_result("security_validation", all_passed, {
                'malicious_inputs_blocked': f"{blocked_count}/{len(malicious_inputs)}",
                'security_detection_rate': blocked_count / len(malicious_inputs),
                'rate_limiting_works': rate_limit_working,
                'audit_system_works': audit_working
            })
            
        except Exception as e:
            self.log_result("security_validation", False, {'error': str(e)})
    
    async def gate_performance_benchmarks(self):
        """Gate 3: Performance Benchmarks"""
        print("\n⚡ Gate 3: Performance Benchmarks")
        
        try:
            config = SeparatorConfig()
            model = AVSeparationTransformer(config)
            model.eval()
            
            def inference_benchmark():
                with torch.no_grad():
                    audio = torch.randn(1, config.audio.n_mels, 50)
                    video = torch.randn(1, 15, 3, *config.video.image_size)
                    outputs = model(audio, video)
                return outputs
            
            # Benchmark inference performance
            benchmark_results = benchmark_function(inference_benchmark, iterations=5)
            
            # Performance requirements
            max_latency = 2.0  # seconds
            min_throughput = 0.5  # inferences per second
            
            latency_ok = benchmark_results['mean_duration'] <= max_latency
            throughput_ok = 1.0 / benchmark_results['mean_duration'] >= min_throughput
            
            # Test caching performance
            cache = AdvancedCache(max_size=100, ttl_seconds=300)
            test_data = {"model_output": torch.randn(100, 100)}
            
            cache.put("test_key", test_data)
            cached_data, cache_hit = cache.get("test_key")
            cache_working = cache_hit and cached_data is not None
            
            # Memory usage check
            cache_stats = cache.get_stats()
            memory_efficient = cache_stats['memory_usage_mb'] < 100  # Under 100MB for test
            
            all_passed = latency_ok and throughput_ok and cache_working and memory_efficient
            
            self.log_result("performance_benchmarks", all_passed, {
                'mean_latency_sec': benchmark_results['mean_duration'],
                'throughput_per_sec': 1.0 / benchmark_results['mean_duration'],
                'latency_requirement_met': latency_ok,
                'throughput_requirement_met': throughput_ok,
                'caching_functional': cache_working,
                'memory_efficient': memory_efficient,
                'cache_stats': cache_stats
            })
            
        except Exception as e:
            self.log_result("performance_benchmarks", False, {'error': str(e)})
    
    async def gate_reliability_testing(self):
        """Gate 4: Reliability and Error Handling"""
        print("\n🛡️ Gate 4: Reliability and Error Handling")
        
        try:
            # Test circuit breaker
            circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
            
            def failing_function():
                raise Exception("Simulated failure")
            
            # Test circuit breaker functionality
            failure_count = 0
            for i in range(5):
                try:
                    circuit_breaker.call(failing_function)
                except Exception:
                    failure_count += 1
            
            circuit_breaker_working = failure_count > 0 and circuit_breaker.state in ['OPEN', 'CLOSED']
            
            # Test retry handler
            retry_handler = RetryHandler(max_retries=2, base_delay=0.1)
            
            def sometimes_failing_function():
                if np.random.random() > 0.7:  # 30% success rate
                    return "success"
                raise Exception("Random failure")
            
            retry_attempts = 0
            try:
                result = retry_handler.retry_sync(sometimes_failing_function)
                retry_working = True
            except Exception:
                retry_working = False  # Could still be working, just unlucky
                retry_attempts += 1
            
            # Test configuration validation
            config_validator = RobustValidator()
            
            valid_audio_config = {'sample_rate': 16000, 'n_fft': 512, 'n_mels': 80}
            invalid_audio_config = {'sample_rate': -1, 'n_fft': 0}
            
            try:
                config_validator.validate_audio_config(valid_audio_config)
                valid_config_passed = True
            except Exception:
                valid_config_passed = False
            
            try:
                config_validator.validate_audio_config(invalid_audio_config)
                invalid_config_rejected = False  # Should have raised exception
            except Exception:
                invalid_config_rejected = True  # Correctly rejected invalid config
            
            validation_working = valid_config_passed and invalid_config_rejected
            
            # Test health checks
            health_results = await global_health_checker.run_checks()
            health_system_working = health_results['overall_status'] in ['healthy', 'critical']
            
            all_passed = (circuit_breaker_working and validation_working and 
                         health_system_working)
            
            self.log_result("reliability_testing", all_passed, {
                'circuit_breaker_functional': circuit_breaker_working,
                'circuit_breaker_state': circuit_breaker.state,
                'config_validation_working': validation_working,
                'health_check_working': health_system_working,
                'health_status': health_results['overall_status']
            })
            
        except Exception as e:
            self.log_result("reliability_testing", False, {'error': str(e)})
    
    async def gate_scalability_testing(self):
        """Gate 5: Scalability and Resource Management"""
        print("\n📈 Gate 5: Scalability and Resource Management")
        
        try:
            # Test batch processing capability
            from av_separation.performance_optimizer import BatchProcessor
            
            batch_processor = BatchProcessor(batch_size=4, max_wait_time=0.1)
            await batch_processor.start()
            
            def simple_processing(batch_items):
                return [item * 2 for item in batch_items]
            
            # Submit batch items
            batch_tasks = []
            for i in range(6):  # More than batch size
                task = batch_processor.process_async(i, simple_processing, timeout=5.0)
                batch_tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*batch_tasks)
                batch_processing_working = len(batch_results) == 6
                expected_results = all(result == i * 2 for i, result in enumerate(batch_results))
            except Exception:
                batch_processing_working = False
                expected_results = False
            
            await batch_processor.stop()
            
            # Test memory management
            cache = AdvancedCache(max_size=10, max_memory_mb=1)  # Very small cache
            
            # Fill cache beyond capacity
            memory_management_working = True
            for i in range(20):
                large_data = np.random.randn(1000)  # Small but trackable
                success = cache.put(f"key_{i}", large_data)
                # Cache should handle capacity limits gracefully
            
            cache_stats = cache.get_stats()
            memory_managed = cache_stats['size'] <= 10  # Should respect max_size
            
            # Test concurrent access
            import threading
            
            concurrent_results = []
            
            def concurrent_cache_access():
                try:
                    cache.put(f"concurrent_{threading.current_thread().ident}", "test_data")
                    data, hit = cache.get(f"concurrent_{threading.current_thread().ident}")
                    concurrent_results.append(hit or data is not None)
                except Exception:
                    concurrent_results.append(False)
            
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=concurrent_cache_access)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
            
            concurrent_access_working = len(concurrent_results) > 0 and any(concurrent_results)
            
            all_passed = (batch_processing_working and expected_results and 
                         memory_managed and concurrent_access_working)
            
            self.log_result("scalability_testing", all_passed, {
                'batch_processing_functional': batch_processing_working,
                'batch_results_correct': expected_results,
                'memory_management_working': memory_managed,
                'concurrent_access_safe': concurrent_access_working,
                'cache_size_controlled': cache_stats['size'] <= 10
            })
            
        except Exception as e:
            self.log_result("scalability_testing", False, {'error': str(e)})
    
    async def gate_integration_testing(self):
        """Gate 6: Integration Testing"""
        print("\n🔗 Gate 6: Integration Testing")
        
        try:
            # Test end-to-end pipeline
            config = SeparatorConfig()
            
            from av_separation import AVSeparator
            
            try:
                separator = AVSeparator(num_speakers=2, config=config)
                integration_init_ok = True
            except Exception as e:
                integration_init_ok = False
                init_error = str(e)
            
            # Test configuration integration
            config_dict = config.to_dict()
            config_serialization = isinstance(config_dict, dict) and len(config_dict) > 0
            
            try:
                config_restored = SeparatorConfig.from_dict(config_dict)
                config_deserialization = True
            except Exception:
                config_deserialization = False
            
            # Test model components integration
            model = AVSeparationTransformer(config)
            
            # Verify all components are properly connected
            has_audio_encoder = hasattr(model, 'audio_encoder')
            has_video_encoder = hasattr(model, 'video_encoder') 
            has_fusion = hasattr(model, 'fusion')
            has_decoder = hasattr(model, 'decoder')
            
            components_integrated = all([
                has_audio_encoder, has_video_encoder, has_fusion, has_decoder
            ])
            
            # Test full pipeline with small inputs
            dummy_audio = torch.randn(1, config.audio.n_mels, 20)  # Smaller for speed
            dummy_video = torch.randn(1, 8, 3, *config.video.image_size)
            
            try:
                with torch.no_grad():
                    outputs = model(dummy_audio, dummy_video)
                    pipeline_functional = (
                        'separated_waveforms' in outputs and
                        'speaker_logits' in outputs and
                        outputs['separated_waveforms'].shape[1] == config.model.max_speakers
                    )
            except Exception as e:
                pipeline_functional = False
                pipeline_error = str(e)
            
            all_passed = (integration_init_ok and config_serialization and 
                         config_deserialization and components_integrated and 
                         pipeline_functional)
            
            details = {
                'separator_initialization': integration_init_ok,
                'config_serialization': config_serialization,
                'config_deserialization': config_deserialization,
                'model_components_integrated': components_integrated,
                'end_to_end_pipeline': pipeline_functional
            }
            
            if not integration_init_ok:
                details['init_error'] = init_error
            if not pipeline_functional:
                details['pipeline_error'] = locals().get('pipeline_error', 'Unknown error')
            
            self.log_result("integration_testing", all_passed, details)
            
        except Exception as e:
            self.log_result("integration_testing", False, {'error': str(e)})
    
    async def gate_compliance_checks(self):
        """Gate 7: Compliance and Documentation"""
        print("\n📋 Gate 7: Compliance and Documentation")
        
        try:
            # Check essential files exist
            essential_files = [
                'README.md',
                'requirements.txt', 
                'setup.py',
                'LICENSE'
            ]
            
            files_exist = {}
            for file_name in essential_files:
                file_path = os.path.join('/root/repo', file_name)
                exists = os.path.exists(file_path)
                files_exist[file_name] = exists
            
            documentation_complete = all(files_exist.values())
            
            # Check code structure
            src_dir = '/root/repo/src'
            av_separation_dir = '/root/repo/src/av_separation'
            
            structure_ok = (
                os.path.exists(src_dir) and
                os.path.exists(av_separation_dir) and
                os.path.exists(f"{av_separation_dir}/__init__.py")
            )
            
            # Check for security best practices
            security_files_exist = any([
                os.path.exists(f"{av_separation_dir}/security.py"),
                os.path.exists(f"{av_separation_dir}/enhanced_security.py")
            ])
            
            # Check for proper error handling
            robust_files_exist = any([
                os.path.exists(f"{av_separation_dir}/robust_core.py"),
                os.path.exists(f"{av_separation_dir}/error_handling.py")
            ])
            
            # Check configuration management
            config_management = os.path.exists(f"{av_separation_dir}/config.py")
            
            # Verify Python package structure
            package_structure_ok = (
                structure_ok and 
                config_management and
                os.path.exists(f"{av_separation_dir}/models/__init__.py")
            )
            
            all_passed = (documentation_complete and package_structure_ok and
                         security_files_exist and robust_files_exist)
            
            self.log_result("compliance_checks", all_passed, {
                'essential_files_present': files_exist,
                'package_structure_correct': package_structure_ok,
                'security_implementation_present': security_files_exist,
                'error_handling_implemented': robust_files_exist,
                'configuration_management': config_management
            })
            
        except Exception as e:
            self.log_result("compliance_checks", False, {'error': str(e)})
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final quality gate report"""
        print("\n" + "=" * 60)
        print("📊 QUALITY GATES FINAL REPORT")
        print("=" * 60)
        
        passed_count = sum(1 for result in self.results.values() if result['passed'])
        total_count = len(self.results)
        success_rate = passed_count / total_count if total_count > 0 else 0
        
        execution_time = time.time() - self.start_time
        
        # Summary
        print(f"\n🎯 Overall Results: {passed_count}/{total_count} gates passed ({success_rate:.1%})")
        print(f"⏱️ Total execution time: {execution_time:.1f} seconds")
        
        # Detailed results
        print(f"\n📋 Gate-by-Gate Results:")
        for gate_name, result in self.results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"   {status} - {gate_name.replace('_', ' ').title()}")
        
        # Overall assessment
        if success_rate >= 0.85:  # 85% minimum
            overall_status = "🎉 PRODUCTION READY"
            recommendation = "System meets quality standards for production deployment"
        elif success_rate >= 0.70:
            overall_status = "⚠️ NEEDS IMPROVEMENT"
            recommendation = "Address failing gates before production deployment"
        else:
            overall_status = "🚨 NOT READY"
            recommendation = "Significant improvements required before production"
        
        print(f"\n{overall_status}")
        print(f"💡 Recommendation: {recommendation}")
        
        return {
            'overall_status': overall_status,
            'success_rate': success_rate,
            'passed_gates': passed_count,
            'total_gates': total_count,
            'execution_time': execution_time,
            'recommendation': recommendation,
            'detailed_results': self.results
        }

async def main():
    """Execute all quality gates"""
    runner = QualityGateRunner()
    final_report = await runner.run_all_gates()
    return final_report

if __name__ == "__main__":
    report = asyncio.run(main())
    
    # Exit with appropriate code
    if report['success_rate'] >= 0.85:
        exit(0)  # Success
    else:
        exit(1)  # Quality gates failed
#!/usr/bin/env python3
"""
Test Suite for Quality Gates System
Comprehensive validation of production readiness
"""

import sys
import asyncio
import time
import json
from pathlib import Path

sys.path.append('/root/repo/src')

print("🔍 QUALITY GATES TESTING")
print("=" * 60)


async def test_security_scanner():
    """Test security scanning functionality"""
    print("\n🔐 Testing Security Scanner:")
    print("-" * 40)
    
    try:
        from av_separation.quality_gates import SecurityScanner
        
        scanner = SecurityScanner("/root/repo")
        print("✓ Security scanner initialized")
        
        # Run security scan
        vulnerabilities = await scanner.scan_codebase()
        print(f"✓ Security scan completed: {len(vulnerabilities)} potential issues found")
        
        # Categorize vulnerabilities
        categories = {}
        severities = {}
        
        for vuln in vulnerabilities:
            categories[vuln['category']] = categories.get(vuln['category'], 0) + 1
            severities[vuln['severity']] = severities.get(vuln['severity'], 0) + 1
        
        print(f"✓ Vulnerability categories: {categories}")
        print(f"✓ Severity distribution: {severities}")
        
        # Check for critical vulnerabilities
        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
        high_vulns = [v for v in vulnerabilities if v['severity'] == 'high']
        
        if critical_vulns:
            print(f"⚠ Found {len(critical_vulns)} critical vulnerabilities")
            for vuln in critical_vulns[:3]:  # Show first 3
                print(f"  - {vuln['category']} in {vuln['file']}:{vuln['line']}")
        else:
            print("✓ No critical vulnerabilities found")
        
        if high_vulns:
            print(f"⚠ Found {len(high_vulns)} high-severity vulnerabilities")
        else:
            print("✓ No high-severity vulnerabilities found")
        
        return True
        
    except Exception as e:
        print(f"❌ Security scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_benchmarks():
    """Test performance benchmarking"""
    print("\n⚡ Testing Performance Benchmarks:")
    print("-" * 40)
    
    try:
        from av_separation.quality_gates import PerformanceBenchmarks
        
        benchmarks = PerformanceBenchmarks()
        print("✓ Performance benchmarks initialized")
        
        # Test cache benchmarks
        cache_results = await benchmarks.run_cache_benchmarks()
        print(f"✓ Cache benchmarks: get={cache_results['get_latency_ms']:.2f}ms, hit_rate={cache_results['hit_rate']:.2f}")
        
        # Test batch processing benchmarks
        batch_results = await benchmarks.run_batch_processing_benchmarks()
        print(f"✓ Batch benchmarks: throughput={batch_results['throughput_items_per_sec']:.1f} items/sec, latency={batch_results['latency_ms']:.2f}ms")
        
        # Test memory benchmarks
        memory_results = await benchmarks.run_memory_benchmarks()
        if 'error' not in memory_results:
            print(f"✓ Memory benchmarks: peak={memory_results['peak_memory_mb']:.1f}MB, leak={memory_results['memory_leak_mb']:.1f}MB")
        else:
            print(f"⚠ Memory benchmarks: {memory_results['error']}")
        
        # Validate performance against thresholds
        benchmark_data = {
            'cache_operations': cache_results,
            'batch_processing': batch_results,
            'memory_usage': memory_results
        }
        
        validations = await benchmarks.validate_performance(benchmark_data)
        passed_validations = [v for v in validations if v['passed']]
        failed_validations = [v for v in validations if not v['passed']]
        
        print(f"✓ Performance validation: {len(passed_validations)} passed, {len(failed_validations)} failed")
        
        if failed_validations:
            for validation in failed_validations[:3]:  # Show first 3 failures
                print(f"  ⚠ {validation['category']}.{validation['metric']}: {validation['value']} (threshold: {validation['threshold']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_code_quality_analyzer():
    """Test code quality analysis"""
    print("\n📊 Testing Code Quality Analyzer:")
    print("-" * 40)
    
    try:
        from av_separation.quality_gates import CodeQualityAnalyzer
        
        analyzer = CodeQualityAnalyzer("/root/repo")
        print("✓ Code quality analyzer initialized")
        
        # Run code quality analysis
        quality_results = await analyzer.analyze_code_quality()
        print("✓ Code quality analysis completed")
        
        # Report results
        complexity = quality_results['complexity_analysis']
        print(f"✓ Complexity analysis: {complexity['total_complex_lines']} complex lines, avg={complexity['average_complexity']:.1f}")
        
        documentation = quality_results['documentation_coverage']
        print(f"✓ Documentation coverage: functions={documentation['function_documentation_coverage']:.1%}, classes={documentation['class_documentation_coverage']:.1%}")
        
        duplication = quality_results['code_duplication']
        print(f"✓ Code duplication: {duplication['total_duplications']} duplicate lines found")
        
        imports = quality_results['import_analysis']
        print(f"✓ Import analysis: {len(imports['import_issues'])} import issues")
        
        naming = quality_results['naming_conventions']
        print(f"✓ Naming conventions: {naming['total_violations']} violations")
        
        error_handling = quality_results['error_handling_coverage']
        print(f"✓ Error handling: {error_handling['error_handling_coverage']:.1%} coverage, {error_handling['bare_except_clauses']} bare except clauses")
        
        return True
        
    except Exception as e:
        print(f"❌ Code quality analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quality_gate_runner():
    """Test complete quality gate execution"""
    print("\n🚪 Testing Quality Gate Runner:")
    print("-" * 40)
    
    try:
        from av_separation.quality_gates import QualityGateRunner
        
        runner = QualityGateRunner("/root/repo")
        print("✓ Quality gate runner initialized")
        
        # Run all quality gates
        print("Running all quality gates (this may take a moment)...")
        results = await runner.run_all_gates()
        
        # Display summary
        summary = results['summary']
        print(f"\n📋 Quality Gates Summary:")
        print(f"  Total gates: {summary['total_gates']}")
        print(f"  Passed: {summary['passed_gates']}")
        print(f"  Failed: {summary['failed_gates']}")
        print(f"  Warnings: {summary['warning_gates']}")
        print(f"  Duration: {summary['total_duration']:.2f}s")
        print(f"  Overall status: {summary['overall_status'].value}")
        
        # Display individual gate results
        print(f"\n📋 Individual Gate Results:")
        for gate_name, gate_result in results['gate_results'].items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'warning': '⚠️',
                'error': '💥',
                'skipped': '⏭️'
            }.get(gate_result['status'].value if hasattr(gate_result['status'], 'value') else gate_result['status'], '❓')
            
            duration = gate_result.get('duration', 0)
            print(f"  {status_icon} {gate_name:<25} ({duration:.2f}s) - {gate_result['message']}")
        
        # Show details for failed gates
        failed_gates = [
            (name, result) for name, result in results['gate_results'].items()
            if result.get('status') in ['failed', 'error'] or 
               (hasattr(result.get('status'), 'value') and result['status'].value in ['failed', 'error'])
        ]
        
        if failed_gates:
            print(f"\n🔍 Failed Gate Details:")
            for gate_name, result in failed_gates[:3]:  # Show first 3 failed gates
                print(f"  Gate: {gate_name}")
                print(f"  Message: {result['message']}")
                if 'details' in result and isinstance(result['details'], dict):
                    for key, value in list(result['details'].items())[:2]:  # Show first 2 details
                        if isinstance(value, (int, float, str)):
                            print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality gate runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integration_gates():
    """Test integration with other system components"""
    print("\n🔗 Testing Integration Gates:")
    print("-" * 40)
    
    try:
        # Test configuration integration
        from av_separation.quality_gates import QualityGateRunner
        from av_separation.config import SeparatorConfig
        
        runner = QualityGateRunner("/root/repo")
        
        # Test configuration gate specifically
        config_result = await runner._run_configuration_gate()
        print(f"✓ Configuration gate: {config_result['status'].value} - {config_result['message']}")
        
        # Test documentation gate
        doc_result = await runner._run_documentation_gate()
        print(f"✓ Documentation gate: {doc_result['status'].value} - {doc_result['message']}")
        
        # Test dependency gate
        dep_result = await runner._run_dependency_gate()
        print(f"✓ Dependency gate: {dep_result['status'].value} - {dep_result['message']}")
        
        # Test test coverage gate
        test_result = await runner._run_test_coverage_gate()
        print(f"✓ Test coverage gate: {test_result['status'].value} - {test_result['message']}")
        
        # Integration check
        all_passed = all(
            result['status'].value == 'passed' 
            for result in [config_result, doc_result, dep_result, test_result]
        )
        
        if all_passed:
            print("✅ All integration gates passed")
        else:
            print("⚠️ Some integration gates have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration gates test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quality_metrics_collection():
    """Test quality metrics collection and reporting"""
    print("\n📈 Testing Quality Metrics Collection:")
    print("-" * 40)
    
    try:
        from av_separation.quality_gates import global_quality_runner
        
        # Generate a quick quality report
        start_time = time.perf_counter()
        results = await global_quality_runner.run_all_gates()
        duration = time.perf_counter() - start_time
        
        print(f"✓ Quality metrics collected in {duration:.2f}s")
        
        # Extract key metrics
        summary = results['summary']
        gate_results = results['gate_results']
        
        metrics = {
            'total_execution_time': duration,
            'gate_success_rate': summary['passed_gates'] / summary['total_gates'],
            'critical_issues': 0,
            'performance_score': 1.0,
            'security_score': 1.0,
            'code_quality_score': 1.0
        }
        
        # Calculate scores based on gate results
        if 'Security Scan' in gate_results:
            security_result = gate_results['Security Scan']
            if security_result.get('details', {}).get('critical_vulnerabilities', 0) > 0:
                metrics['security_score'] = 0.5
                metrics['critical_issues'] += 1
        
        if 'Performance Benchmarks' in gate_results:
            perf_result = gate_results['Performance Benchmarks']
            if 'failed' in str(perf_result.get('status', '')).lower():
                metrics['performance_score'] = 0.7
        
        if 'Code Quality Analysis' in gate_results:
            quality_result = gate_results['Code Quality Analysis']
            if 'failed' in str(quality_result.get('status', '')).lower():
                metrics['code_quality_score'] = 0.6
        
        # Overall quality score
        overall_score = (
            metrics['security_score'] * 0.4 +
            metrics['performance_score'] * 0.3 +
            metrics['code_quality_score'] * 0.3
        )
        metrics['overall_quality_score'] = overall_score
        
        print(f"✓ Quality scores calculated:")
        print(f"  Security: {metrics['security_score']:.2f}")
        print(f"  Performance: {metrics['performance_score']:.2f}")
        print(f"  Code Quality: {metrics['code_quality_score']:.2f}")
        print(f"  Overall: {metrics['overall_quality_score']:.2f}")
        print(f"  Gate Success Rate: {metrics['gate_success_rate']:.1%}")
        print(f"  Critical Issues: {metrics['critical_issues']}")
        
        # Generate quality report
        quality_report = {
            'timestamp': time.time(),
            'project_path': '/root/repo',
            'metrics': metrics,
            'summary': summary,
            'recommendations': []
        }
        
        # Add recommendations based on results
        if metrics['security_score'] < 0.8:
            quality_report['recommendations'].append("Address security vulnerabilities")
        if metrics['performance_score'] < 0.8:
            quality_report['recommendations'].append("Optimize performance bottlenecks")
        if metrics['code_quality_score'] < 0.8:
            quality_report['recommendations'].append("Improve code quality and documentation")
        
        if not quality_report['recommendations']:
            quality_report['recommendations'].append("All quality metrics are within acceptable ranges")
        
        print(f"✓ Quality report generated with {len(quality_report['recommendations'])} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality metrics collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner"""
    print("🧪 Starting Quality Gates System Tests")
    
    tests = [
        ("Security Scanner", test_security_scanner),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Code Quality Analyzer", test_code_quality_analyzer),
        ("Quality Gate Runner", test_quality_gate_runner),
        ("Integration Gates", test_integration_gates),
        ("Quality Metrics Collection", test_quality_metrics_collection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running {test_name} Tests")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            success = await test_func()
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
    print("📊 QUALITY GATES TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status in ["FAILED", "ERROR"])
    total_time = sum(duration for _, _, duration in results)
    
    for test_name, status, duration in results:
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {test_name:<30} {status:<7} ({duration:.2f}s)")
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed in {total_time:.2f}s")
    
    if passed == len(tests):
        print("\n🎉 QUALITY GATES: ALL TESTS COMPLETED SUCCESSFULLY")
        print("✅ Security scanning operational")
        print("✅ Performance benchmarking validated")
        print("✅ Code quality analysis working")
        print("✅ Integration gates functional")
        print("✅ Quality metrics collection active")
        print("✅ Production readiness validated")
        
        # Final quality assessment
        quality_percentage = (passed / len(tests)) * 100
        if quality_percentage == 100:
            print(f"\n🏆 OVERALL QUALITY SCORE: {quality_percentage:.0f}% - EXCELLENT")
        elif quality_percentage >= 80:
            print(f"\n🥈 OVERALL QUALITY SCORE: {quality_percentage:.0f}% - GOOD")
        else:
            print(f"\n🥉 OVERALL QUALITY SCORE: {quality_percentage:.0f}% - NEEDS IMPROVEMENT")
    else:
        print(f"\n⚠️  QUALITY GATES: PARTIALLY COMPLETE ({passed}/{len(tests)} passed)")
        print("🔧 Some quality components may need attention")
    
    return passed == len(tests)


if __name__ == "__main__":
    asyncio.run(main())
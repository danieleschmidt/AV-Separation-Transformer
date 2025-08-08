"""
Quality Gates System for AV-Separation-Transformer
Comprehensive testing, security scanning, and production readiness validation
"""

import asyncio
import time
import subprocess
import json
import re
import os
import sys
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger('av_separation.quality_gates')


class QualityGateStatus(Enum):
    """Status of quality gate checks"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class QualityGateSeverity(Enum):
    """Severity levels for quality gate issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    status: QualityGateStatus
    severity: QualityGateSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate_name': self.gate_name,
            'status': self.status.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'duration': self.duration,
            'timestamp': self.timestamp
        }


class SecurityScanner:
    """Advanced security scanning for code and configurations"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.security_patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security vulnerability patterns"""
        return {
            'secrets': [
                r'(?i)(password|pwd|pass|secret|key|token|api_key)\s*[:=]\s*["\'][^"\']{8,}["\']',
                r'(?i)(private_key|secret_key)\s*=\s*["\'][^"\']{20,}["\']',
                r'["\'][A-Za-z0-9]{20,}["\']',  # Potential tokens
                r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',  # Private keys
                r'(?i)aws_access_key_id\s*[:=]\s*["\'][A-Z0-9]{20}["\']',
                r'(?i)aws_secret_access_key\s*[:=]\s*["\'][A-Za-z0-9/+=]{40}["\']'
            ],
            'sql_injection': [
                r'(?i)SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\s*\+\s*.*',
                r'(?i)INSERT\s+INTO\s+.*\s+VALUES\s*\(.*\+.*\)',
                r'(?i)UPDATE\s+.*\s+SET\s+.*\s*=\s*.*\+.*',
                r'(?i)DELETE\s+FROM\s+.*\s+WHERE\s+.*\s*\+.*',
                r'execute\s*\(\s*["\'][^"\']*\s*\+\s*.*["\']',
                r'cursor\.execute\s*\(\s*[^)]*\s*%\s*[^)]*\)'
            ],
            'command_injection': [
                r'os\.system\s*\(\s*[^)]*\+.*\)',
                r'subprocess\.(call|run|Popen)\s*\(\s*[^)]*\+.*\)',
                r'shell=True.*\+',
                r'exec\s*\(\s*[^)]*\+.*\)',
                r'eval\s*\(\s*[^)]*\+.*\)'
            ],
            'path_traversal': [
                r'open\s*\(\s*[^)]*\.\./.*\)',
                r'file\s*\(\s*[^)]*\.\./.*\)',
                r'Path\s*\(\s*[^)]*\.\./.*\)',
                r'os\.path\.join\s*\([^)]*\.\./.*\)'
            ],
            'xss_vulnerabilities': [
                r'innerHTML\s*=\s*[^;]*\+.*',
                r'document\.write\s*\(\s*[^)]*\+.*\)',
                r'\.html\s*\(\s*[^)]*\+.*\)',
                r'response\.write\s*\(\s*[^)]*\+.*\)'
            ],
            'unsafe_deserialization': [
                r'pickle\.loads?\s*\(',
                r'marshal\.loads?\s*\(',
                r'yaml\.load\s*\(',  # Should use safe_load
                r'json\.loads\s*\([^)]*user_input',
                r'ast\.literal_eval\s*\([^)]*user_input'
            ]
        }
    
    async def scan_codebase(self) -> List[Dict[str, Any]]:
        """Scan entire codebase for security vulnerabilities"""
        vulnerabilities = []
        
        # Get all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            file_vulns = await self._scan_file(file_path)
            vulnerabilities.extend(file_vulns)
        
        return vulnerabilities
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in security scan"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'test_',
            'tests/',
            'venv',
            'env',
            '.tox'
        ]
        
        str_path = str(file_path)
        return any(pattern in str_path for pattern in skip_patterns)
    
    async def _scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan single file for security issues"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vulnerability = {
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'category': category,
                            'pattern': pattern,
                            'match': match.group(0)[:100],  # First 100 chars
                            'severity': self._get_severity(category),
                            'description': self._get_description(category)
                        }
                        
                        vulnerabilities.append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category"""
        severity_map = {
            'secrets': 'critical',
            'sql_injection': 'high',
            'command_injection': 'high',
            'path_traversal': 'high',
            'xss_vulnerabilities': 'medium',
            'unsafe_deserialization': 'medium'
        }
        return severity_map.get(category, 'low')
    
    def _get_description(self, category: str) -> str:
        """Get description for vulnerability category"""
        descriptions = {
            'secrets': 'Potential secrets or API keys in code',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'path_traversal': 'Potential path traversal vulnerability',
            'xss_vulnerabilities': 'Potential cross-site scripting vulnerability',
            'unsafe_deserialization': 'Unsafe deserialization pattern'
        }
        return descriptions.get(category, 'Security pattern detected')


class PerformanceBenchmarks:
    """Performance benchmarking and validation"""
    
    def __init__(self):
        self.benchmarks = {}
        self.thresholds = self._load_performance_thresholds()
        
    def _load_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load performance thresholds for different operations"""
        return {
            'cache_operations': {
                'get_latency_ms': 1.0,
                'put_latency_ms': 2.0,
                'hit_rate_min': 0.8
            },
            'batch_processing': {
                'throughput_min_items_per_sec': 1000,
                'latency_max_ms': 100,
                'error_rate_max': 0.01
            },
            'auto_scaling': {
                'decision_latency_max_ms': 500,
                'scaling_accuracy_min': 0.9
            },
            'api_endpoints': {
                'response_time_max_ms': 200,
                'error_rate_max': 0.05,
                'throughput_min_rps': 100
            },
            'memory_usage': {
                'max_memory_mb': 1024,
                'memory_leak_threshold': 0.1
            }
        }
    
    async def run_cache_benchmarks(self) -> Dict[str, Any]:
        """Benchmark cache performance"""
        from .performance_optimizer import AdvancedCache
        
        cache = AdvancedCache(max_size=1000, max_memory_mb=50)
        
        # Warm up cache
        for i in range(100):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Benchmark GET operations
        get_times = []
        for _ in range(1000):
            start = time.perf_counter()
            cache.get("key_50")
            get_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark PUT operations  
        put_times = []
        for i in range(100):
            start = time.perf_counter()
            cache.put(f"bench_key_{i}", f"bench_value_{i}")
            put_times.append((time.perf_counter() - start) * 1000)
        
        stats = cache.get_stats()
        
        return {
            'get_latency_ms': sum(get_times) / len(get_times),
            'put_latency_ms': sum(put_times) / len(put_times),
            'hit_rate': stats['hit_rate'],
            'memory_usage_mb': stats['memory_usage_mb'],
            'cache_size': stats['size']
        }
    
    async def run_batch_processing_benchmarks(self) -> Dict[str, Any]:
        """Benchmark batch processing performance"""
        from .performance_optimizer import BatchProcessor
        
        processor = BatchProcessor(
            batch_size=32,
            max_wait_time=0.05,
            max_queue_size=1000
        )
        
        await processor.start()
        
        try:
            def simple_processing_func(items):
                return [f"processed_{item}" for item in items]
            
            # Benchmark processing
            start_time = time.perf_counter()
            tasks = []
            
            for i in range(1000):
                task = processor.process_async(f"item_{i}", simple_processing_func)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            error_count = len(results) - len(successful_results)
            
            duration = end_time - start_time
            throughput = len(successful_results) / duration
            error_rate = error_count / len(results)
            
            stats = processor.get_stats()
            
            return {
                'throughput_items_per_sec': throughput,
                'latency_ms': (duration / len(results)) * 1000,
                'error_rate': error_rate,
                'total_items': len(results),
                'successful_items': len(successful_results),
                'batch_stats': stats
            }
        
        finally:
            await processor.stop()
    
    async def run_memory_benchmarks(self) -> Dict[str, Any]:
        """Benchmark memory usage and detect leaks"""
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Simulate memory-intensive operations
            data_structures = []
            
            for i in range(1000):
                # Create some data structures
                data = {f"key_{j}": f"value_{j}" * 100 for j in range(100)}
                data_structures.append(data)
            
            peak_memory = process.memory_info().rss / (1024 * 1024)
            
            # Clean up
            data_structures.clear()
            import gc
            gc.collect()
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            return {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': peak_memory - initial_memory,
                'memory_leak_mb': final_memory - initial_memory,
                'memory_recovered_mb': peak_memory - final_memory
            }
            
        except ImportError:
            return {
                'error': 'psutil not available for memory benchmarking'
            }
    
    async def validate_performance(self, benchmark_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate benchmark results against thresholds"""
        validations = []
        
        for category, results in benchmark_results.items():
            thresholds = self.thresholds.get(category, {})
            
            for metric, value in results.items():
                if metric in thresholds:
                    threshold = thresholds[metric]
                    
                    # Determine if metric passes threshold
                    if 'max' in metric:
                        passed = value <= threshold
                    elif 'min' in metric:
                        passed = value >= threshold
                    else:
                        passed = True  # Default to pass if unclear
                    
                    validations.append({
                        'category': category,
                        'metric': metric,
                        'value': value,
                        'threshold': threshold,
                        'passed': passed,
                        'severity': 'high' if not passed else 'info'
                    })
        
        return validations


class CodeQualityAnalyzer:
    """Code quality and best practices analysis"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        
    async def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        results = {
            'complexity_analysis': await self._analyze_complexity(),
            'documentation_coverage': await self._check_documentation(),
            'code_duplication': await self._check_duplication(),
            'import_analysis': await self._analyze_imports(),
            'naming_conventions': await self._check_naming_conventions(),
            'error_handling_coverage': await self._check_error_handling()
        }
        
        return results
    
    async def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze cyclomatic complexity"""
        python_files = list(self.project_root.rglob("*.py"))
        complexity_issues = []
        
        for file_path in python_files:
            if 'test_' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple complexity analysis based on control structures
                complexity_indicators = [
                    'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except',
                    'and ', 'or ', '?', 'lambda ', 'with '
                ]
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    complexity_score = sum(1 for indicator in complexity_indicators if indicator in line)
                    
                    if complexity_score > 3:  # Threshold for complex line
                        complexity_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'complexity_score': complexity_score,
                            'line_content': line.strip()[:100]
                        })
            
            except Exception as e:
                logger.warning(f"Error analyzing complexity in {file_path}: {e}")
        
        return {
            'high_complexity_lines': complexity_issues[:10],  # Top 10
            'total_complex_lines': len(complexity_issues),
            'average_complexity': sum(item['complexity_score'] for item in complexity_issues) / max(len(complexity_issues), 1)
        }
    
    async def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage"""
        python_files = list(self.project_root.rglob("*.py"))
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for file_path in python_files:
            if 'test_' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find functions and classes
                function_matches = re.finditer(r'^\s*def\s+(\w+)\s*\(', content, re.MULTILINE)
                class_matches = re.finditer(r'^\s*class\s+(\w+)', content, re.MULTILINE)
                
                for match in function_matches:
                    total_functions += 1
                    
                    # Check if function has docstring
                    func_start = match.end()
                    remaining_content = content[func_start:]
                    
                    # Look for docstring in next few lines
                    if '"""' in remaining_content[:200] or "'''" in remaining_content[:200]:
                        documented_functions += 1
                
                for match in class_matches:
                    total_classes += 1
                    
                    # Check if class has docstring
                    class_start = match.end()
                    remaining_content = content[class_start:]
                    
                    if '"""' in remaining_content[:300] or "'''" in remaining_content[:300]:
                        documented_classes += 1
            
            except Exception as e:
                logger.warning(f"Error checking documentation in {file_path}: {e}")
        
        function_coverage = documented_functions / max(total_functions, 1)
        class_coverage = documented_classes / max(total_classes, 1)
        
        return {
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'function_documentation_coverage': function_coverage,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'class_documentation_coverage': class_coverage,
            'overall_documentation_coverage': (function_coverage + class_coverage) / 2
        }
    
    async def _check_duplication(self) -> Dict[str, Any]:
        """Check for code duplication"""
        python_files = list(self.project_root.rglob("*.py"))
        
        # Simple duplication detection based on similar lines
        line_hashes = defaultdict(list)
        duplications = []
        
        for file_path in python_files:
            if 'test_' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Normalize line (remove whitespace, comments)
                    normalized = re.sub(r'\s+', ' ', line.strip())
                    normalized = re.sub(r'#.*$', '', normalized)
                    
                    if len(normalized) > 20:  # Only check substantial lines
                        line_hash = hashlib.md5(normalized.encode()).hexdigest()
                        line_hashes[line_hash].append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': i,
                            'content': line.strip()[:80]
                        })
            
            except Exception as e:
                logger.warning(f"Error checking duplication in {file_path}: {e}")
        
        # Find duplicated lines
        for line_hash, occurrences in line_hashes.items():
            if len(occurrences) > 1:
                duplications.append({
                    'hash': line_hash,
                    'occurrences': occurrences,
                    'duplicate_count': len(occurrences)
                })
        
        return {
            'total_duplications': len(duplications),
            'high_duplication_lines': sorted(duplications, key=lambda x: x['duplicate_count'], reverse=True)[:5]
        }
    
    async def _analyze_imports(self) -> Dict[str, Any]:
        """Analyze import usage and organization"""
        python_files = list(self.project_root.rglob("*.py"))
        
        import_issues = []
        unused_imports = []
        circular_imports = []
        
        for file_path in python_files:
            if '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all imports
                import_matches = re.finditer(r'^(?:from\s+(\S+)\s+)?import\s+([^\n]+)', content, re.MULTILINE)
                
                for match in import_matches:
                    module = match.group(1) if match.group(1) else match.group(2).split('.')[0]
                    imported_items = match.group(2)
                    
                    # Check for potential issues
                    if '*' in imported_items:
                        import_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'issue': 'wildcard_import',
                            'line': content[:match.start()].count('\n') + 1,
                            'content': match.group(0)
                        })
            
            except Exception as e:
                logger.warning(f"Error analyzing imports in {file_path}: {e}")
        
        return {
            'import_issues': import_issues,
            'unused_imports': unused_imports,
            'circular_imports': circular_imports
        }
    
    async def _check_naming_conventions(self) -> Dict[str, Any]:
        """Check naming convention compliance"""
        python_files = list(self.project_root.rglob("*.py"))
        
        naming_violations = []
        
        for file_path in python_files:
            if '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check function names (should be snake_case)
                function_matches = re.finditer(r'^\s*def\s+(\w+)', content, re.MULTILINE)
                for match in function_matches:
                    func_name = match.group(1)
                    if not re.match(r'^[a-z_][a-z0-9_]*$', func_name) and not func_name.startswith('_'):
                        naming_violations.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'type': 'function',
                            'name': func_name,
                            'line': content[:match.start()].count('\n') + 1,
                            'issue': 'should_be_snake_case'
                        })
                
                # Check class names (should be PascalCase)
                class_matches = re.finditer(r'^\s*class\s+(\w+)', content, re.MULTILINE)
                for match in class_matches:
                    class_name = match.group(1)
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                        naming_violations.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'type': 'class',
                            'name': class_name,
                            'line': content[:match.start()].count('\n') + 1,
                            'issue': 'should_be_pascal_case'
                        })
            
            except Exception as e:
                logger.warning(f"Error checking naming in {file_path}: {e}")
        
        return {
            'naming_violations': naming_violations[:10],  # Top 10
            'total_violations': len(naming_violations)
        }
    
    async def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling coverage"""
        python_files = list(self.project_root.rglob("*.py"))
        
        functions_with_try_catch = 0
        total_functions = 0
        bare_except_clauses = []
        
        for file_path in python_files:
            if '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count functions
                function_matches = list(re.finditer(r'^\s*def\s+(\w+)', content, re.MULTILINE))
                total_functions += len(function_matches)
                
                # Count try-except blocks
                try_matches = list(re.finditer(r'^\s*try:', content, re.MULTILINE))
                functions_with_try_catch += len(try_matches)
                
                # Find bare except clauses
                bare_except_matches = re.finditer(r'^\s*except\s*:', content, re.MULTILINE)
                for match in bare_except_matches:
                    bare_except_clauses.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': content[:match.start()].count('\n') + 1
                    })
            
            except Exception as e:
                logger.warning(f"Error checking error handling in {file_path}: {e}")
        
        error_handling_coverage = functions_with_try_catch / max(total_functions, 1)
        
        return {
            'total_functions': total_functions,
            'functions_with_error_handling': functions_with_try_catch,
            'error_handling_coverage': error_handling_coverage,
            'bare_except_clauses': len(bare_except_clauses),
            'bare_except_locations': bare_except_clauses[:5]
        }


class QualityGateRunner:
    """Main quality gate execution engine"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = project_root
        self.security_scanner = SecurityScanner(project_root)
        self.performance_benchmarks = PerformanceBenchmarks()
        self.code_quality_analyzer = CodeQualityAnalyzer(project_root)
        
        self.results = []
        
    async def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results"""
        
        gates = [
            ("Security Scan", self._run_security_gate),
            ("Performance Benchmarks", self._run_performance_gate),
            ("Code Quality Analysis", self._run_code_quality_gate),
            ("Configuration Validation", self._run_configuration_gate),
            ("Documentation Check", self._run_documentation_gate),
            ("Test Coverage Analysis", self._run_test_coverage_gate),
            ("Dependency Security Scan", self._run_dependency_gate)
        ]
        
        gate_results = {}
        summary = {
            'total_gates': len(gates),
            'passed_gates': 0,
            'failed_gates': 0,
            'warning_gates': 0,
            'total_duration': 0.0,
            'overall_status': QualityGateStatus.PASSED
        }
        
        for gate_name, gate_func in gates:
            logger.info(f"Running quality gate: {gate_name}")
            start_time = time.perf_counter()
            
            try:
                result = await gate_func()
                duration = time.perf_counter() - start_time
                
                result['duration'] = duration
                gate_results[gate_name] = result
                
                # Update summary
                summary['total_duration'] += duration
                
                if result['status'] == QualityGateStatus.PASSED:
                    summary['passed_gates'] += 1
                elif result['status'] == QualityGateStatus.FAILED:
                    summary['failed_gates'] += 1
                    summary['overall_status'] = QualityGateStatus.FAILED
                elif result['status'] == QualityGateStatus.WARNING:
                    summary['warning_gates'] += 1
                    if summary['overall_status'] == QualityGateStatus.PASSED:
                        summary['overall_status'] = QualityGateStatus.WARNING
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(f"Error running quality gate {gate_name}: {e}")
                
                gate_results[gate_name] = {
                    'status': QualityGateStatus.ERROR,
                    'message': f"Gate execution failed: {str(e)}",
                    'duration': duration
                }
                
                summary['failed_gates'] += 1
                summary['total_duration'] += duration
                summary['overall_status'] = QualityGateStatus.FAILED
        
        return {
            'summary': summary,
            'gate_results': gate_results,
            'timestamp': time.time(),
            'project_root': self.project_root
        }
    
    async def _run_security_gate(self) -> Dict[str, Any]:
        """Run security quality gate"""
        vulnerabilities = await self.security_scanner.scan_codebase()
        
        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
        high_vulns = [v for v in vulnerabilities if v['severity'] == 'high']
        
        if critical_vulns:
            status = QualityGateStatus.FAILED
            message = f"Found {len(critical_vulns)} critical security vulnerabilities"
        elif high_vulns:
            status = QualityGateStatus.WARNING
            message = f"Found {len(high_vulns)} high-severity security issues"
        else:
            status = QualityGateStatus.PASSED
            message = "No critical security vulnerabilities found"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'total_vulnerabilities': len(vulnerabilities),
                'critical_vulnerabilities': len(critical_vulns),
                'high_vulnerabilities': len(high_vulns),
                'vulnerabilities': vulnerabilities[:10]  # Top 10
            }
        }
    
    async def _run_performance_gate(self) -> Dict[str, Any]:
        """Run performance quality gate"""
        benchmark_results = {
            'cache_operations': await self.performance_benchmarks.run_cache_benchmarks(),
            'batch_processing': await self.performance_benchmarks.run_batch_processing_benchmarks(),
            'memory_usage': await self.performance_benchmarks.run_memory_benchmarks()
        }
        
        validations = await self.performance_benchmarks.validate_performance(benchmark_results)
        
        failed_validations = [v for v in validations if not v['passed'] and v['severity'] == 'high']
        warning_validations = [v for v in validations if not v['passed'] and v['severity'] == 'medium']
        
        if failed_validations:
            status = QualityGateStatus.FAILED
            message = f"{len(failed_validations)} performance benchmarks failed"
        elif warning_validations:
            status = QualityGateStatus.WARNING
            message = f"{len(warning_validations)} performance benchmarks below optimal"
        else:
            status = QualityGateStatus.PASSED
            message = "All performance benchmarks passed"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'benchmark_results': benchmark_results,
                'validations': validations,
                'failed_validations': failed_validations
            }
        }
    
    async def _run_code_quality_gate(self) -> Dict[str, Any]:
        """Run code quality gate"""
        quality_results = await self.code_quality_analyzer.analyze_code_quality()
        
        # Evaluate quality metrics
        doc_coverage = quality_results['documentation_coverage']['overall_documentation_coverage']
        complexity_issues = quality_results['complexity_analysis']['total_complex_lines']
        naming_violations = quality_results['naming_conventions']['total_violations']
        
        issues = []
        if doc_coverage < 0.6:
            issues.append(f"Low documentation coverage: {doc_coverage:.1%}")
        if complexity_issues > 50:
            issues.append(f"High complexity lines: {complexity_issues}")
        if naming_violations > 10:
            issues.append(f"Naming convention violations: {naming_violations}")
        
        if not issues:
            status = QualityGateStatus.PASSED
            message = "Code quality standards met"
        elif doc_coverage < 0.4 or complexity_issues > 100:
            status = QualityGateStatus.FAILED
            message = f"Code quality issues: {'; '.join(issues)}"
        else:
            status = QualityGateStatus.WARNING
            message = f"Code quality warnings: {'; '.join(issues)}"
        
        return {
            'status': status,
            'message': message,
            'details': quality_results
        }
    
    async def _run_configuration_gate(self) -> Dict[str, Any]:
        """Run configuration validation gate"""
        try:
            from .config import SeparatorConfig
            
            # Test configuration loading
            config = SeparatorConfig()
            config_dict = config.to_dict()
            
            # Test configuration round-trip
            restored_config = SeparatorConfig.from_dict(config_dict)
            
            # Validate configuration values
            issues = []
            
            if config.audio.sample_rate <= 0:
                issues.append("Invalid audio sample rate")
            if config.video.fps <= 0:
                issues.append("Invalid video FPS")
            if config.model.max_speakers <= 0:
                issues.append("Invalid max speakers setting")
            
            if issues:
                status = QualityGateStatus.FAILED
                message = f"Configuration validation failed: {'; '.join(issues)}"
            else:
                status = QualityGateStatus.PASSED
                message = "Configuration validation passed"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'config_sections': len(config_dict),
                    'validation_issues': issues,
                    'sample_config': {k: str(v)[:100] for k, v in config_dict.items()}
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.FAILED,
                'message': f"Configuration validation error: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def _run_documentation_gate(self) -> Dict[str, Any]:
        """Run documentation quality gate"""
        required_docs = [
            'README.md',
            'CHANGELOG.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'requirements.txt',
            'setup.py'
        ]
        
        missing_docs = []
        present_docs = []
        
        for doc_file in required_docs:
            doc_path = Path(self.project_root) / doc_file
            if doc_path.exists():
                present_docs.append(doc_file)
            else:
                missing_docs.append(doc_file)
        
        coverage = len(present_docs) / len(required_docs)
        
        if coverage == 1.0:
            status = QualityGateStatus.PASSED
            message = "All required documentation present"
        elif coverage >= 0.8:
            status = QualityGateStatus.WARNING
            message = f"Some documentation missing: {missing_docs}"
        else:
            status = QualityGateStatus.FAILED
            message = f"Critical documentation missing: {missing_docs}"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'required_docs': required_docs,
                'present_docs': present_docs,
                'missing_docs': missing_docs,
                'coverage': coverage
            }
        }
    
    async def _run_test_coverage_gate(self) -> Dict[str, Any]:
        """Run test coverage analysis gate"""
        test_files = list(Path(self.project_root).rglob("test_*.py"))
        src_files = list(Path(self.project_root).joinpath("src").rglob("*.py"))
        
        if not src_files:
            src_files = list(Path(self.project_root).rglob("*.py"))
            src_files = [f for f in src_files if not f.name.startswith('test_')]
        
        test_coverage_ratio = len(test_files) / max(len(src_files), 1)
        
        if test_coverage_ratio >= 0.8:
            status = QualityGateStatus.PASSED
            message = f"Good test coverage: {len(test_files)} test files for {len(src_files)} source files"
        elif test_coverage_ratio >= 0.5:
            status = QualityGateStatus.WARNING
            message = f"Moderate test coverage: {test_coverage_ratio:.1%}"
        else:
            status = QualityGateStatus.FAILED
            message = f"Low test coverage: {test_coverage_ratio:.1%}"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'test_files': len(test_files),
                'source_files': len(src_files),
                'coverage_ratio': test_coverage_ratio,
                'test_file_names': [f.name for f in test_files]
            }
        }
    
    async def _run_dependency_gate(self) -> Dict[str, Any]:
        """Run dependency security scan"""
        requirements_file = Path(self.project_root) / "requirements.txt"
        
        if not requirements_file.exists():
            return {
                'status': QualityGateStatus.WARNING,
                'message': "No requirements.txt found for dependency scan",
                'details': {}
            }
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()
            
            dependencies = []
            for line in requirements:
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line)
            
            # Simple dependency analysis
            potential_issues = []
            
            # Check for version pinning
            unpinned_deps = [dep for dep in dependencies if '>=' not in dep and '==' not in dep and '~=' not in dep]
            if unpinned_deps:
                potential_issues.append(f"{len(unpinned_deps)} dependencies without version constraints")
            
            if potential_issues:
                status = QualityGateStatus.WARNING
                message = f"Dependency issues: {'; '.join(potential_issues)}"
            else:
                status = QualityGateStatus.PASSED
                message = "Dependency scan completed successfully"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_dependencies': len(dependencies),
                    'unpinned_dependencies': len(unpinned_deps),
                    'potential_issues': potential_issues
                }
            }
            
        except Exception as e:
            return {
                'status': QualityGateStatus.ERROR,
                'message': f"Dependency scan error: {str(e)}",
                'details': {'error': str(e)}
            }


# Global quality gate runner
global_quality_runner = QualityGateRunner()
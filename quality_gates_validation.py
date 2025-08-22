#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation System
Autonomously validates security, performance, and code quality.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityGateReport:
    """Complete quality gate validation report."""
    overall_passed: bool
    overall_score: float
    results: List[QualityGateResult]
    summary: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration: float = 0.0


class SecurityQualityGate:
    """Security-focused quality gate."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> List[Dict[str, Any]]:
        """Load security vulnerability patterns."""
        return [
            {
                'name': 'hardcoded_secrets',
                'pattern': r'(password|secret|key|token)\s*=\s*["\'][^"\'/\s][^"\'/\s]+["\']',
                'severity': 'high',
                'description': 'Hardcoded secrets detected'
            },
            {
                'name': 'sql_injection',
                'pattern': r'(execute|query)\s*\([^)]*\+[^)]*\)',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability'
            },
            {
                'name': 'path_traversal',
                'pattern': r'(open|read|write)\s*\([^)]*\.\.[/\\]',
                'severity': 'medium',
                'description': 'Potential path traversal vulnerability'
            },
            {
                'name': 'unsafe_deserialization',
                'pattern': r'(pickle\.loads|eval|exec)\s*\(',
                'severity': 'high',
                'description': 'Unsafe deserialization detected'
            },
            {
                'name': 'weak_crypto',
                'pattern': r'(md5|sha1)\s*\(',
                'severity': 'medium',
                'description': 'Weak cryptographic algorithm'
            },
            {
                'name': 'debug_code',
                'pattern': r'(print|console\.log|debugger)\s*\(',
                'severity': 'low',
                'description': 'Debug code left in production'
            }
        ]
    
    def validate(self) -> QualityGateResult:
        """Run security validation."""
        start_time = time.time()
        
        vulnerabilities = []
        files_scanned = 0
        lines_scanned = 0
        
        # Scan Python files
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                files_scanned += 1
                lines_scanned += len(lines)
                
                for line_num, line in enumerate(lines, 1):
                    for pattern_info in self.security_patterns:
                        if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                            vulnerabilities.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'type': pattern_info['name'],
                                'severity': pattern_info['severity'],
                                'description': pattern_info['description'],
                                'code_snippet': line.strip()
                            })
                            
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
        
        # Calculate security score
        high_severity = len([v for v in vulnerabilities if v['severity'] == 'high'])
        medium_severity = len([v for v in vulnerabilities if v['severity'] == 'medium'])
        low_severity = len([v for v in vulnerabilities if v['severity'] == 'low'])
        
        # Weighted scoring
        penalty = high_severity * 0.3 + medium_severity * 0.2 + low_severity * 0.1
        security_score = max(0.0, 1.0 - penalty)
        
        # Generate recommendations
        recommendations = []
        if high_severity > 0:
            recommendations.append(f"Fix {high_severity} high-severity security issues immediately")
        if medium_severity > 0:
            recommendations.append(f"Address {medium_severity} medium-severity security issues")
        if low_severity > 0:
            recommendations.append(f"Clean up {low_severity} low-severity issues for best practices")
        
        if not vulnerabilities:
            recommendations.append("Security scan passed - no vulnerabilities detected")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Security Validation",
            passed=high_severity == 0,  # Pass only if no high-severity issues
            score=security_score,
            details={
                'vulnerabilities': vulnerabilities,
                'files_scanned': files_scanned,
                'lines_scanned': lines_scanned,
                'severity_breakdown': {
                    'high': high_severity,
                    'medium': medium_severity,
                    'low': low_severity
                }
            },
            recommendations=recommendations,
            execution_time=execution_time
        )


class PerformanceQualityGate:
    """Performance-focused quality gate."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def validate(self) -> QualityGateResult:
        """Run performance validation."""
        start_time = time.time()
        
        # Performance analysis
        performance_issues = []
        files_analyzed = 0
        
        # Common performance anti-patterns
        anti_patterns = [
            {
                'name': 'nested_loops',
                'pattern': r'for\s+.*:\s*\n\s*for\s+.*:',
                'severity': 'medium',
                'description': 'Nested loops detected - O(n²) complexity'
            },
            {
                'name': 'inefficient_string_concat',
                'pattern': r'\w+\s*\+=\s*["\'].*["\']',
                'severity': 'low',
                'description': 'Inefficient string concatenation'
            },
            {
                'name': 'unbounded_recursion',
                'pattern': r'def\s+(\w+).*:\s*.*\1\s*\(',
                'severity': 'high',
                'description': 'Potential unbounded recursion'
            }
        ]
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                files_analyzed += 1
                
                for pattern_info in anti_patterns:
                    matches = re.finditer(pattern_info['pattern'], content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        performance_issues.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'type': pattern_info['name'],
                            'severity': pattern_info['severity'],
                            'description': pattern_info['description']
                        })
                        
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        # Calculate performance score
        high_perf_issues = len([i for i in performance_issues if i['severity'] == 'high'])
        medium_perf_issues = len([i for i in performance_issues if i['severity'] == 'medium'])
        low_perf_issues = len([i for i in performance_issues if i['severity'] == 'low'])
        
        penalty = high_perf_issues * 0.4 + medium_perf_issues * 0.2 + low_perf_issues * 0.1
        perf_score = max(0.0, 1.0 - penalty)
        
        # Memory and CPU analysis
        memory_analysis = self._analyze_memory_patterns()
        cpu_analysis = self._analyze_cpu_patterns()
        
        recommendations = []
        if high_perf_issues > 0:
            recommendations.append(f"Critical: Fix {high_perf_issues} high-impact performance issues")
        if medium_perf_issues > 0:
            recommendations.append(f"Optimize {medium_perf_issues} moderate performance bottlenecks")
        if memory_analysis['issues'] > 0:
            recommendations.append(f"Address {memory_analysis['issues']} memory-related concerns")
        if cpu_analysis['issues'] > 0:
            recommendations.append(f"Optimize {cpu_analysis['issues']} CPU-intensive operations")
        
        if not performance_issues and memory_analysis['issues'] == 0:
            recommendations.append("Performance analysis passed - no major issues detected")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Performance Validation",
            passed=high_perf_issues == 0,
            score=perf_score,
            details={
                'performance_issues': performance_issues,
                'files_analyzed': files_analyzed,
                'severity_breakdown': {
                    'high': high_perf_issues,
                    'medium': medium_perf_issues,
                    'low': low_perf_issues
                },
                'memory_analysis': memory_analysis,
                'cpu_analysis': cpu_analysis
            },
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        memory_patterns = [
            r'\[.*for.*in.*for.*in.*\]',  # Nested list comprehensions
            r'np\.zeros\(\d+\*\d+',        # Large numpy allocations
            r'torch\..*\(.*,\s*device=',   # GPU memory allocations
        ]
        
        issues = 0
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                for pattern in memory_patterns:
                    if re.search(pattern, content):
                        issues += 1
            except Exception:
                pass
        
        return {
            'issues': issues,
            'patterns_checked': len(memory_patterns),
            'status': 'good' if issues == 0 else 'needs_attention'
        }
    
    def _analyze_cpu_patterns(self) -> Dict[str, Any]:
        """Analyze CPU-intensive patterns."""
        cpu_patterns = [
            r'while\s+True:',              # Infinite loops
            r'for.*range\(\d{4,}\)',       # Large range loops
            r'time\.sleep\(0\)',           # Busy waiting
        ]
        
        issues = 0
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                for pattern in cpu_patterns:
                    if re.search(pattern, content):
                        issues += 1
            except Exception:
                pass
        
        return {
            'issues': issues,
            'patterns_checked': len(cpu_patterns),
            'status': 'good' if issues == 0 else 'needs_attention'
        }


class CodeQualityGate:
    """Code quality and best practices validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def validate(self) -> QualityGateResult:
        """Run code quality validation."""
        start_time = time.time()
        
        quality_metrics = {
            'documentation_coverage': self._check_documentation_coverage(),
            'type_hints_coverage': self._check_type_hints(),
            'error_handling': self._check_error_handling(),
            'code_complexity': self._analyze_complexity(),
            'naming_conventions': self._check_naming_conventions(),
            'import_organization': self._check_imports()
        }
        
        # Calculate overall quality score
        scores = [metric['score'] for metric in quality_metrics.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Generate recommendations
        recommendations = []
        for metric_name, metric_result in quality_metrics.items():
            if metric_result['score'] < 0.7:  # Below acceptable threshold
                recommendations.extend(metric_result['recommendations'])
        
        if overall_score >= 0.85:
            recommendations.append("Excellent code quality standards maintained")
        elif overall_score >= 0.7:
            recommendations.append("Good code quality with room for improvement")
        else:
            recommendations.append("Code quality needs significant improvement")
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            name="Code Quality Validation",
            passed=overall_score >= 0.7,
            score=overall_score,
            details={
                'metrics': quality_metrics,
                'detailed_analysis': self._get_detailed_analysis()
            },
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    def _check_documentation_coverage(self) -> Dict[str, Any]:
        """Check documentation coverage."""
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Count functions and their docstrings
                func_matches = re.finditer(r'^\s*def\s+(\w+)', content, re.MULTILINE)
                for match in func_matches:
                    total_functions += 1
                    # Check if function has docstring
                    func_start = match.end()
                    remaining_content = content[func_start:]
                    if re.search(r'\s*""".*?"""', remaining_content, re.DOTALL):
                        documented_functions += 1
                
                # Count classes and their docstrings
                class_matches = re.finditer(r'^\s*class\s+(\w+)', content, re.MULTILINE)
                for match in class_matches:
                    total_classes += 1
                    # Check if class has docstring
                    class_start = match.end()
                    remaining_content = content[class_start:]
                    if re.search(r'\s*""".*?"""', remaining_content, re.DOTALL):
                        documented_classes += 1
                        
            except Exception:
                continue
        
        # Calculate coverage
        func_coverage = documented_functions / max(1, total_functions)
        class_coverage = documented_classes / max(1, total_classes)
        overall_coverage = (func_coverage + class_coverage) / 2
        
        recommendations = []
        if func_coverage < 0.8:
            recommendations.append(f"Add docstrings to {total_functions - documented_functions} functions")
        if class_coverage < 0.8:
            recommendations.append(f"Add docstrings to {total_classes - documented_classes} classes")
        
        return {
            'score': overall_coverage,
            'function_coverage': func_coverage,
            'class_coverage': class_coverage,
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'recommendations': recommendations
        }
    
    def _check_type_hints(self) -> Dict[str, Any]:
        """Check type hints coverage."""
        total_functions = 0
        typed_functions = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Find function definitions with type hints
                func_pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:'
                func_matches = re.finditer(func_pattern, content)
                
                for match in func_matches:
                    total_functions += 1
                    func_def = match.group(0)
                    
                    # Check for type hints in parameters or return type
                    if ':' in func_def and ('->' in func_def or ': ' in func_def):
                        typed_functions += 1
                        
            except Exception:
                continue
        
        coverage = typed_functions / max(1, total_functions)
        
        recommendations = []
        if coverage < 0.6:
            recommendations.append(f"Add type hints to {total_functions - typed_functions} functions")
        
        return {
            'score': coverage,
            'total_functions': total_functions,
            'typed_functions': typed_functions,
            'recommendations': recommendations
        }
    
    def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling patterns."""
        total_functions = 0
        functions_with_error_handling = 0
        bare_except_count = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Count functions
                total_functions += len(re.findall(r'^\s*def\s+', content, re.MULTILINE))
                
                # Count functions with try/except blocks
                functions_with_error_handling += len(re.findall(r'def[^{]*{[^}]*try[^}]*except', content, re.DOTALL))
                
                # Count bare except clauses (bad practice)
                bare_except_count += len(re.findall(r'except:\s*$', content, re.MULTILINE))
                
            except Exception:
                continue
        
        error_handling_score = functions_with_error_handling / max(1, total_functions)
        # Penalize bare except clauses
        penalty = min(0.5, bare_except_count * 0.1)
        final_score = max(0.0, error_handling_score - penalty)
        
        recommendations = []
        if error_handling_score < 0.5:
            recommendations.append("Add proper error handling to more functions")
        if bare_except_count > 0:
            recommendations.append(f"Replace {bare_except_count} bare except clauses with specific exceptions")
        
        return {
            'score': final_score,
            'total_functions': total_functions,
            'functions_with_error_handling': functions_with_error_handling,
            'bare_except_count': bare_except_count,
            'recommendations': recommendations
        }
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity."""
        high_complexity_functions = []
        total_functions = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Simple complexity analysis based on control flow statements
                func_matches = re.finditer(r'^(\s*)def\s+(\w+)', content, re.MULTILINE)
                
                for match in func_matches:
                    total_functions += 1
                    func_name = match.group(2)
                    indent_level = len(match.group(1))
                    
                    # Find function body (simplified)
                    func_start = match.end()
                    lines = content[func_start:].split('\n')
                    
                    complexity = 1  # Base complexity
                    for line in lines:
                        if len(line.strip()) == 0:
                            continue
                        
                        # If we've reached a function at the same or lower indentation, stop
                        current_indent = len(line) - len(line.lstrip())
                        if current_indent <= indent_level and line.strip().startswith(('def ', 'class ')):
                            break
                        
                        # Count complexity-increasing statements
                        complexity_keywords = ['if', 'elif', 'for', 'while', 'except', 'and', 'or']
                        for keyword in complexity_keywords:
                            if re.search(rf'\b{keyword}\b', line):
                                complexity += 1
                    
                    if complexity > 10:  # High complexity threshold
                        high_complexity_functions.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'function': func_name,
                            'complexity': complexity
                        })
                        
            except Exception:
                continue
        
        # Score based on percentage of high-complexity functions
        high_complexity_ratio = len(high_complexity_functions) / max(1, total_functions)
        complexity_score = max(0.0, 1.0 - high_complexity_ratio * 2)  # Penalize heavily
        
        recommendations = []
        if len(high_complexity_functions) > 0:
            recommendations.append(f"Refactor {len(high_complexity_functions)} high-complexity functions")
            recommendations.extend([
                f"Consider breaking down {func['function']} in {func['file']} (complexity: {func['complexity']})"
                for func in high_complexity_functions[:3]  # Top 3
            ])
        
        return {
            'score': complexity_score,
            'total_functions': total_functions,
            'high_complexity_functions': len(high_complexity_functions),
            'high_complexity_details': high_complexity_functions,
            'recommendations': recommendations
        }
    
    def _check_naming_conventions(self) -> Dict[str, Any]:
        """Check naming convention compliance."""
        violations = []
        total_identifiers = 0
        
        naming_patterns = {
            'function': (r'def\s+(\w+)', r'^[a-z][a-z0-9_]*$'),
            'class': (r'class\s+(\w+)', r'^[A-Z][A-Za-z0-9]*$'),
            'constant': (r'^\s*([A-Z][A-Z0-9_]+)\s*=', r'^[A-Z][A-Z0-9_]*$'),
        }
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                for identifier_type, (find_pattern, naming_pattern) in naming_patterns.items():
                    matches = re.finditer(find_pattern, content, re.MULTILINE)
                    
                    for match in matches:
                        total_identifiers += 1
                        identifier = match.group(1)
                        
                        if not re.match(naming_pattern, identifier):
                            violations.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'type': identifier_type,
                                'name': identifier,
                                'line': content[:match.start()].count('\n') + 1
                            })
                            
            except Exception:
                continue
        
        # Score based on naming convention compliance
        compliance_rate = 1.0 - (len(violations) / max(1, total_identifiers))
        
        recommendations = []
        if len(violations) > 0:
            recommendations.append(f"Fix {len(violations)} naming convention violations")
            
            # Group violations by type
            violations_by_type = {}
            for violation in violations:
                vtype = violation['type']
                if vtype not in violations_by_type:
                    violations_by_type[vtype] = []
                violations_by_type[vtype].append(violation)
            
            for vtype, vlist in violations_by_type.items():
                recommendations.append(f"Fix {len(vlist)} {vtype} naming violations")
        
        return {
            'score': compliance_rate,
            'total_identifiers': total_identifiers,
            'violations': len(violations),
            'violation_details': violations[:10],  # Top 10
            'recommendations': recommendations
        }
    
    def _check_imports(self) -> Dict[str, Any]:
        """Check import organization and best practices."""
        import_issues = []
        total_files = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            total_files += 1
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                # Check import organization
                imports_section = []
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith(('import ', 'from ')):
                        imports_section.append((i, stripped))
                    elif stripped and not stripped.startswith('#') and imports_section:
                        # Non-import line after imports - imports should be at top
                        break
                
                # Check for wildcard imports
                for line_num, import_line in imports_section:
                    if 'import *' in import_line:
                        import_issues.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num + 1,
                            'issue': 'wildcard_import',
                            'description': 'Wildcard import detected'
                        })
                    
                    # Check for unused imports (basic check)
                    if import_line.startswith('import '):
                        module = import_line.split()[1].split('.')[0]
                        if module not in content[sum(len(l) + 1 for l in lines[:10]):]:  # Skip import section
                            import_issues.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num + 1,
                                'issue': 'possibly_unused_import',
                                'description': f'Possibly unused import: {module}'
                            })
                            
            except Exception:
                continue
        
        # Score based on import issues
        issues_per_file = len(import_issues) / max(1, total_files)
        import_score = max(0.0, 1.0 - issues_per_file)
        
        recommendations = []
        if len(import_issues) > 0:
            recommendations.append(f"Fix {len(import_issues)} import-related issues")
            
            wildcard_count = len([i for i in import_issues if i['issue'] == 'wildcard_import'])
            unused_count = len([i for i in import_issues if i['issue'] == 'possibly_unused_import'])
            
            if wildcard_count > 0:
                recommendations.append(f"Replace {wildcard_count} wildcard imports with specific imports")
            if unused_count > 0:
                recommendations.append(f"Remove {unused_count} possibly unused imports")
        
        return {
            'score': import_score,
            'total_files': total_files,
            'import_issues': len(import_issues),
            'issue_details': import_issues[:10],  # Top 10
            'recommendations': recommendations
        }
    
    def _get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed code analysis."""
        total_lines = 0
        total_files = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if 'test' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            total_files += 1
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                total_lines += len(content.split('\n'))
            except Exception:
                continue
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'average_lines_per_file': total_lines / max(1, total_files)
        }


class QualityGateValidator:
    """Main quality gate validation orchestrator."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.gates = [
            SecurityQualityGate(self.project_root),
            PerformanceQualityGate(self.project_root),
            CodeQualityGate(self.project_root)
        ]
    
    def validate_all(self, parallel: bool = True) -> QualityGateReport:
        """Run all quality gates."""
        start_time = time.time()
        
        if parallel:
            results = self._validate_parallel()
        else:
            results = self._validate_sequential()
        
        # Calculate overall metrics
        overall_score = sum(result.score for result in results) / len(results)
        overall_passed = all(result.passed for result in results)
        
        # Generate summary
        summary = {
            'total_gates': len(results),
            'passed_gates': sum(1 for r in results if r.passed),
            'failed_gates': sum(1 for r in results if not r.passed),
            'average_score': overall_score,
            'min_score': min(r.score for r in results),
            'max_score': max(r.score for r in results),
            'total_recommendations': sum(len(r.recommendations) for r in results)
        }
        
        duration = time.time() - start_time
        
        return QualityGateReport(
            overall_passed=overall_passed,
            overall_score=overall_score,
            results=results,
            summary=summary,
            duration=duration
        )
    
    def _validate_parallel(self) -> List[QualityGateResult]:
        """Run quality gates in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(self.gates)) as executor:
            future_to_gate = {executor.submit(gate.validate): gate for gate in self.gates}
            
            for future in as_completed(future_to_gate):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per gate
                    results.append(result)
                except Exception as e:
                    gate = future_to_gate[future]
                    print(f"Error running {gate.__class__.__name__}: {e}")
                    # Create a failed result
                    results.append(QualityGateResult(
                        name=f"{gate.__class__.__name__} (Failed)",
                        passed=False,
                        score=0.0,
                        details={'error': str(e)},
                        recommendations=[f"Fix {gate.__class__.__name__} execution error"]
                    ))
        
        return results
    
    def _validate_sequential(self) -> List[QualityGateResult]:
        """Run quality gates sequentially."""
        results = []
        
        for gate in self.gates:
            try:
                result = gate.validate()
                results.append(result)
            except Exception as e:
                print(f"Error running {gate.__class__.__name__}: {e}")
                results.append(QualityGateResult(
                    name=f"{gate.__class__.__name__} (Failed)",
                    passed=False,
                    score=0.0,
                    details={'error': str(e)},
                    recommendations=[f"Fix {gate.__class__.__name__} execution error"]
                ))
        
        return results
    
    def generate_report(self, report: QualityGateReport, output_file: Optional[Path] = None) -> str:
        """Generate a comprehensive quality gate report."""
        # Create detailed report
        report_content = self._create_detailed_report(report)
        
        if output_file:
            output_file.write_text(report_content, encoding='utf-8')
            print(f"Quality gate report saved to: {output_file}")
        
        return report_content
    
    def _create_detailed_report(self, report: QualityGateReport) -> str:
        """Create detailed quality gate report."""
        lines = [
            "# 🛡️ Quality Gate Validation Report",
            "",
            f"**Report Generated:** {report.timestamp}",
            f"**Validation Duration:** {report.duration:.2f} seconds",
            f"**Overall Status:** {'✅ PASSED' if report.overall_passed else '❌ FAILED'}",
            f"**Overall Score:** {report.overall_score:.2f}/1.00 ({report.overall_score*100:.1f}%)",
            "",
            "## Summary",
            "",
            f"- **Total Gates:** {report.summary['total_gates']}",
            f"- **Passed:** {report.summary['passed_gates']}",
            f"- **Failed:** {report.summary['failed_gates']}",
            f"- **Score Range:** {report.summary['min_score']:.2f} - {report.summary['max_score']:.2f}",
            f"- **Total Recommendations:** {report.summary['total_recommendations']}",
            "",
            "## Gate Results",
            ""
        ]
        
        for result in report.results:
            status_emoji = "✅" if result.passed else "❌"
            lines.extend([
                f"### {status_emoji} {result.name}",
                "",
                f"- **Status:** {'PASSED' if result.passed else 'FAILED'}",
                f"- **Score:** {result.score:.2f}/1.00 ({result.score*100:.1f}%)",
                f"- **Execution Time:** {result.execution_time:.2f} seconds",
                "",
                "**Details:**",
            ])
            
            # Add relevant details based on gate type
            if 'vulnerabilities' in result.details:
                vuln_count = len(result.details['vulnerabilities'])
                lines.append(f"- Vulnerabilities Found: {vuln_count}")
                
                if vuln_count > 0:
                    severity_breakdown = result.details.get('severity_breakdown', {})
                    lines.extend([
                        f"  - High Severity: {severity_breakdown.get('high', 0)}",
                        f"  - Medium Severity: {severity_breakdown.get('medium', 0)}",
                        f"  - Low Severity: {severity_breakdown.get('low', 0)}"
                    ])
            
            if 'performance_issues' in result.details:
                issue_count = len(result.details['performance_issues'])
                lines.append(f"- Performance Issues: {issue_count}")
            
            if 'metrics' in result.details:
                metrics = result.details['metrics']
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'score' in metric_data:
                        lines.append(f"- {metric_name.replace('_', ' ').title()}: {metric_data['score']:.2f}")
            
            if result.recommendations:
                lines.extend([
                    "",
                    "**Recommendations:**"
                ])
                for rec in result.recommendations:
                    lines.append(f"- {rec}")
            
            lines.extend(["", "---", ""])
        
        # Add final recommendations
        if not report.overall_passed:
            lines.extend([
                "## 🚨 Action Required",
                "",
                "The quality gates have failed. Please address the following:",
                ""
            ])
            
            failed_gates = [r for r in report.results if not r.passed]
            for gate in failed_gates:
                lines.append(f"### {gate.name}")
                for rec in gate.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        else:
            lines.extend([
                "## 🎉 Quality Gates Passed!",
                "",
                "All quality gates have passed successfully. The system meets the required standards for:",
                "- Security",
                "- Performance",
                "- Code Quality",
                "",
                "The system is ready for production deployment."
            ])
        
        return "\n".join(lines)


def main():
    """Main entry point for quality gate validation."""
    print("🛡️ Starting Comprehensive Quality Gate Validation")
    print("="*60)
    
    project_root = Path.cwd()
    validator = QualityGateValidator(project_root)
    
    # Run validation
    print("Running quality gates...")
    report = validator.validate_all(parallel=True)
    
    # Generate and save report
    report_file = project_root / "quality_gate_report.md"
    validator.generate_report(report, report_file)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Quality Gate Validation {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
    print(f"Overall Score: {report.overall_score:.2f}/1.00 ({report.overall_score*100:.1f}%)")
    print(f"Duration: {report.duration:.2f} seconds")
    
    if not report.overall_passed:
        print("\n🚨 Action Required:")
        failed_gates = [r for r in report.results if not r.passed]
        for gate in failed_gates:
            print(f"  - {gate.name}: {gate.score:.2f} score")
        print(f"\nDetailed report saved to: {report_file}")
        return 1
    else:
        print("\n🎉 All quality gates passed! System ready for deployment.")
        return 0


if __name__ == '__main__':
    exit(main())

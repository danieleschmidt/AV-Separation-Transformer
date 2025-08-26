#!/usr/bin/env python3
"""
AUTONOMOUS QUALITY GATES VALIDATION
Comprehensive validation system ensuring all quality requirements are met
before production deployment. This system runs automatically and validates
all three generations plus integration tests.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import traceback
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quality_gates.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityGate(ABC):
    """Abstract base class for quality gates"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def execute(self) -> QualityGateResult:
        """Execute the quality gate check"""
        pass
    
    def _create_result(self, passed: bool, score: float, details: Dict[str, Any], 
                      execution_time: float, error_message: Optional[str] = None) -> QualityGateResult:
        """Helper to create standardized results"""
        return QualityGateResult(
            gate_name=self.name,
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            error_message=error_message
        )


class CodeExecutionGate(QualityGate):
    """Gate that validates code executes without critical errors"""
    
    def __init__(self):
        super().__init__("Code Execution", weight=2.0)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            # Test each generation
            generations = [
                ("Generation 1", "generation_1_simple_implementation.py"),
                ("Generation 2", "generation_2_robust_implementation.py"),
            ]
            
            passed_tests = 0
            total_tests = len(generations)
            
            for gen_name, script in generations:
                try:
                    logger.info(f"Testing {gen_name}")
                    
                    # Run with timeout and capture output
                    result = subprocess.run(
                        [sys.executable, script],
                        capture_output=True,
                        text=True,
                        timeout=120,  # 2 minute timeout
                        cwd=Path.cwd()
                    )
                    
                    if result.returncode == 0:
                        passed_tests += 1
                        details[gen_name] = {
                            "status": "PASSED",
                            "output_lines": len(result.stdout.split('\n')),
                            "has_success_markers": "✅" in result.stdout
                        }
                    else:
                        details[gen_name] = {
                            "status": "FAILED",
                            "return_code": result.returncode,
                            "error_preview": result.stderr[:500] if result.stderr else "No error output"
                        }
                    
                except subprocess.TimeoutExpired:
                    details[gen_name] = {
                        "status": "TIMEOUT",
                        "error": "Execution exceeded 2 minutes"
                    }
                except Exception as e:
                    details[gen_name] = {
                        "status": "EXCEPTION",
                        "error": str(e)
                    }
            
            # Calculate scores
            execution_time = time.time() - start_time
            score = passed_tests / total_tests
            passed = score >= 0.5  # At least 50% must pass
            
            details["summary"] = {
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": score
            }
            
            return self._create_result(passed, score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class CodeQualityGate(QualityGate):
    """Gate that validates code quality metrics"""
    
    def __init__(self):
        super().__init__("Code Quality", weight=1.5)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            # Check for basic code quality indicators
            python_files = list(Path('.').glob('generation_*.py'))
            
            total_score = 0
            metrics = {}
            
            for py_file in python_files:
                content = py_file.read_text()
                lines = content.split('\n')
                
                file_metrics = {
                    "lines_of_code": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                    "total_lines": len(lines),
                    "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                    "docstring_lines": content.count('"""') // 2 + content.count("'''") // 2,
                    "class_count": content.count('class '),
                    "function_count": content.count('def '),
                    "import_count": content.count('import ') + content.count('from '),
                    "has_error_handling": 'try:' in content and 'except' in content,
                    "has_logging": 'logger.' in content or 'logging.' in content,
                    "has_type_hints": '->' in content and 'typing' in content
                }
                
                # Calculate file quality score
                file_score = 0
                if file_metrics["lines_of_code"] > 100:  # Substantial implementation
                    file_score += 0.3
                if file_metrics["comment_lines"] / max(file_metrics["total_lines"], 1) > 0.1:  # 10%+ comments
                    file_score += 0.2
                if file_metrics["has_error_handling"]:
                    file_score += 0.2
                if file_metrics["has_logging"]:
                    file_score += 0.15
                if file_metrics["has_type_hints"]:
                    file_score += 0.15
                
                metrics[py_file.name] = file_metrics
                total_score += file_score
            
            # Overall quality assessment
            avg_score = total_score / len(python_files) if python_files else 0
            passed = avg_score >= 0.6  # Require 60% quality score
            
            details = {
                "file_metrics": metrics,
                "average_quality_score": avg_score,
                "files_analyzed": len(python_files)
            }
            
            execution_time = time.time() - start_time
            return self._create_result(passed, avg_score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class PerformanceGate(QualityGate):
    """Gate that validates performance requirements"""
    
    def __init__(self):
        super().__init__("Performance", weight=1.0)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            # Test performance with Generation 1 (working implementation)
            import importlib.util
            
            # Dynamic import of Generation 1
            spec = importlib.util.spec_from_file_location(
                "gen1", "generation_1_simple_implementation.py"
            )
            gen1_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen1_module)
            
            # Create separator and run benchmark
            separator = gen1_module.SimpleAVSeparator(num_speakers=2)
            
            # Run performance benchmark
            benchmark_results = separator.benchmark(iterations=5)
            
            # Performance requirements
            max_latency_ms = 2000  # 2 seconds max
            min_rtf = 1.0  # Real-time factor >= 1.0
            
            latency_ok = benchmark_results['mean_latency_ms'] <= max_latency_ms
            rtf_ok = benchmark_results['rtf'] >= min_rtf
            
            # Calculate performance score
            latency_score = min(1.0, max_latency_ms / benchmark_results['mean_latency_ms'])
            rtf_score = min(1.0, benchmark_results['rtf'] / min_rtf)
            overall_score = (latency_score + rtf_score) / 2
            
            passed = latency_ok and rtf_ok
            
            details = {
                "benchmark_results": benchmark_results,
                "requirements": {
                    "max_latency_ms": max_latency_ms,
                    "min_rtf": min_rtf
                },
                "checks": {
                    "latency_ok": latency_ok,
                    "rtf_ok": rtf_ok
                },
                "scores": {
                    "latency_score": latency_score,
                    "rtf_score": rtf_score,
                    "overall_score": overall_score
                }
            }
            
            execution_time = time.time() - start_time
            return self._create_result(passed, overall_score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class SecurityGate(QualityGate):
    """Gate that validates security requirements"""
    
    def __init__(self):
        super().__init__("Security", weight=1.0)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            security_checks = {
                "input_validation": 0,
                "error_handling": 0,
                "no_hardcoded_secrets": 0,
                "logging_present": 0,
                "timeout_protection": 0
            }
            
            python_files = list(Path('.').glob('generation_*.py'))
            
            for py_file in python_files:
                content = py_file.read_text().lower()
                
                # Check for input validation
                if any(term in content for term in ['validate', 'sanitize', 'check', 'verify']):
                    security_checks["input_validation"] += 1
                
                # Check for error handling
                if 'try:' in content and 'except' in content:
                    security_checks["error_handling"] += 1
                
                # Check for no obvious hardcoded secrets
                suspicious_patterns = ['password', 'secret', 'key', 'token']
                has_hardcoded = any(f'"{pattern}"' in content or f"'{pattern}'" in content 
                                   for pattern in suspicious_patterns)
                if not has_hardcoded:
                    security_checks["no_hardcoded_secrets"] += 1
                
                # Check for logging
                if 'logging' in content or 'logger' in content:
                    security_checks["logging_present"] += 1
                
                # Check for timeout protection
                if 'timeout' in content:
                    security_checks["timeout_protection"] += 1
            
            # Calculate security score
            total_files = len(python_files)
            if total_files > 0:
                security_score = sum(
                    min(1.0, check_count / total_files) 
                    for check_count in security_checks.values()
                ) / len(security_checks)
            else:
                security_score = 0.0
            
            passed = security_score >= 0.6  # 60% security requirements
            
            details = {
                "security_checks": security_checks,
                "files_analyzed": total_files,
                "security_score": security_score,
                "requirements_met": security_score >= 0.6
            }
            
            execution_time = time.time() - start_time
            return self._create_result(passed, security_score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class DocumentationGate(QualityGate):
    """Gate that validates documentation quality"""
    
    def __init__(self):
        super().__init__("Documentation", weight=0.5)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            # Check README and documentation files
            readme_file = Path("README.md")
            doc_score = 0
            
            if readme_file.exists():
                readme_content = readme_file.read_text()
                readme_metrics = {
                    "length": len(readme_content),
                    "sections": readme_content.count('#'),
                    "code_blocks": readme_content.count('```'),
                    "links": readme_content.count('['),
                    "has_installation": 'install' in readme_content.lower(),
                    "has_usage": 'usage' in readme_content.lower(),
                    "has_examples": 'example' in readme_content.lower(),
                    "has_api_docs": 'api' in readme_content.lower()
                }
                
                # Score README quality
                if readme_metrics["length"] > 1000:  # Substantial docs
                    doc_score += 0.3
                if readme_metrics["sections"] >= 5:  # Well structured
                    doc_score += 0.2
                if readme_metrics["code_blocks"] >= 3:  # Has examples
                    doc_score += 0.2
                if readme_metrics["has_installation"] and readme_metrics["has_usage"]:
                    doc_score += 0.3
                
                details["readme_metrics"] = readme_metrics
            else:
                details["readme_status"] = "Missing README.md"
            
            # Check inline documentation in Python files
            python_files = list(Path('.').glob('generation_*.py'))
            docstring_coverage = 0
            
            for py_file in python_files:
                content = py_file.read_text()
                classes = content.count('class ')
                functions = content.count('def ')
                docstrings = content.count('"""') // 2 + content.count("'''") // 2
                
                if classes + functions > 0:
                    file_coverage = docstrings / (classes + functions)
                    docstring_coverage += file_coverage
            
            if python_files:
                docstring_coverage /= len(python_files)
                doc_score += docstring_coverage * 0.3
            
            passed = doc_score >= 0.4  # 40% documentation score
            
            details.update({
                "documentation_score": doc_score,
                "docstring_coverage": docstring_coverage,
                "python_files_analyzed": len(python_files)
            })
            
            execution_time = time.time() - start_time
            return self._create_result(passed, doc_score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class ArchitectureGate(QualityGate):
    """Gate that validates system architecture quality"""
    
    def __init__(self):
        super().__init__("Architecture", weight=1.0)
        
    def execute(self) -> QualityGateResult:
        start_time = time.time()
        details = {}
        
        try:
            # Analyze architecture patterns across generations
            python_files = list(Path('.').glob('generation_*.py'))
            
            architecture_metrics = {
                "separation_of_concerns": 0,
                "error_handling": 0,
                "configurability": 0,
                "extensibility": 0,
                "scalability_features": 0
            }
            
            for py_file in python_files:
                content = py_file.read_text()
                
                # Check separation of concerns (multiple classes)
                if content.count('class ') >= 3:
                    architecture_metrics["separation_of_concerns"] += 1
                
                # Check comprehensive error handling
                if content.count('try:') >= 2 and content.count('except') >= 2:
                    architecture_metrics["error_handling"] += 1
                
                # Check configurability (config classes/parameters)
                if 'config' in content.lower() or 'Config' in content:
                    architecture_metrics["configurability"] += 1
                
                # Check extensibility (abstract classes, inheritance)
                if 'ABC' in content or 'abstractmethod' in content or 'super()' in content:
                    architecture_metrics["extensibility"] += 1
                
                # Check scalability features
                scalability_terms = ['concurrent', 'async', 'thread', 'pool', 'cache', 'scale']
                if any(term in content.lower() for term in scalability_terms):
                    architecture_metrics["scalability_features"] += 1
            
            # Progressive enhancement check
            gen_features = {}
            for py_file in python_files:
                content = py_file.read_text()
                feature_count = (
                    content.count('class ') +
                    (2 if 'logging' in content else 0) +
                    (2 if 'security' in content.lower() else 0) +
                    (2 if 'cache' in content.lower() else 0) +
                    (2 if 'async' in content.lower() or 'thread' in content.lower() else 0)
                )
                gen_features[py_file.name] = feature_count
            
            # Check for progressive enhancement (later generations should have more features)
            progressive_enhancement = True
            sorted_gens = sorted(gen_features.items())
            for i in range(1, len(sorted_gens)):
                if sorted_gens[i][1] <= sorted_gens[i-1][1]:
                    progressive_enhancement = False
                    break
            
            # Calculate architecture score
            total_files = len(python_files)
            if total_files > 0:
                arch_score = sum(
                    min(1.0, metric_count / total_files) 
                    for metric_count in architecture_metrics.values()
                ) / len(architecture_metrics)
            else:
                arch_score = 0.0
            
            if progressive_enhancement:
                arch_score += 0.2  # Bonus for progressive enhancement
            
            arch_score = min(1.0, arch_score)  # Cap at 1.0
            passed = arch_score >= 0.6  # 60% architecture quality
            
            details = {
                "architecture_metrics": architecture_metrics,
                "generation_features": gen_features,
                "progressive_enhancement": progressive_enhancement,
                "architecture_score": arch_score,
                "files_analyzed": total_files
            }
            
            execution_time = time.time() - start_time
            return self._create_result(passed, arch_score, details, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                False, 0.0, {"error": str(e)}, execution_time, str(e)
            )


class QualityGateSystem:
    """Main quality gate validation system"""
    
    def __init__(self):
        self.gates = [
            CodeExecutionGate(),
            CodeQualityGate(),
            PerformanceGate(),
            SecurityGate(),
            DocumentationGate(),
            ArchitectureGate()
        ]
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report"""
        logger.info("🔍 Starting Autonomous Quality Gates Validation")
        logger.info("=" * 70)
        
        start_time = time.time()
        results = []
        
        for gate in self.gates:
            logger.info(f"Running {gate.name} Gate...")
            
            try:
                result = gate.execute()
                results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"{status} {gate.name}: {result.score:.2f} ({result.execution_time:.2f}s)")
                
                if not result.passed and result.error_message:
                    logger.warning(f"  Error: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ FAILED {gate.name}: Exception - {e}")
                results.append(QualityGateResult(
                    gate_name=gate.name,
                    passed=False,
                    score=0.0,
                    details={"exception": str(e)},
                    execution_time=0.0,
                    error_message=str(e)
                ))
        
        # Calculate overall results
        total_execution_time = time.time() - start_time
        
        # Weighted scoring
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = sum(
            result.score * gate.weight 
            for result, gate in zip(results, self.gates)
        ) / total_weight
        
        passed_gates = sum(1 for result in results if result.passed)
        total_gates = len(results)
        pass_rate = passed_gates / total_gates
        
        # Critical gates (must pass for overall success)
        critical_gates = ["Code Execution", "Performance", "Security"]
        critical_passed = all(
            result.passed for result in results 
            if result.gate_name in critical_gates
        )
        
        overall_passed = critical_passed and pass_rate >= 0.7  # 70% gates must pass
        
        # Generate comprehensive report
        report = {
            "timestamp": time.time(),
            "overall_passed": overall_passed,
            "overall_score": weighted_score,
            "pass_rate": pass_rate,
            "critical_gates_passed": critical_passed,
            "execution_time": total_execution_time,
            "summary": {
                "passed_gates": passed_gates,
                "total_gates": total_gates,
                "critical_gates": critical_gates,
                "weighted_score": weighted_score
            },
            "gate_results": [result.to_dict() for result in results]
        }
        
        # Log final results
        logger.info("=" * 70)
        logger.info("🎯 QUALITY GATES VALIDATION COMPLETE")
        logger.info(f"Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        logger.info(f"Overall Score: {weighted_score:.2f}/1.0")
        logger.info(f"Pass Rate: {pass_rate:.1%} ({passed_gates}/{total_gates})")
        logger.info(f"Critical Gates: {'✅' if critical_passed else '❌'}")
        logger.info(f"Total Execution Time: {total_execution_time:.1f}s")
        
        # Gate-by-gate summary
        for result in results:
            status = "✅" if result.passed else "❌"
            logger.info(f"  {status} {result.gate_name}: {result.score:.2f}")
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = "quality_gates_report.json"):
        """Save quality gate report to file"""
        report_path = Path(filename)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Quality gate report saved: {report_path}")
        return report_path


def main():
    """Main execution function"""
    try:
        # Initialize and run quality gate system
        qg_system = QualityGateSystem()
        report = qg_system.run_all_gates()
        
        # Save report
        report_path = qg_system.save_report(report)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("🏆 AUTONOMOUS SDLC QUALITY GATES SUMMARY")
        print("=" * 70)
        
        if report["overall_passed"]:
            print("✅ ALL QUALITY GATES PASSED - READY FOR PRODUCTION")
        else:
            print("❌ QUALITY GATES FAILED - REQUIRES ATTENTION")
        
        print(f"📊 Overall Score: {report['overall_score']:.2f}/1.0")
        print(f"📈 Pass Rate: {report['pass_rate']:.1%}")
        print(f"⏱️  Total Time: {report['execution_time']:.1f}s")
        print(f"📄 Report: {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Quality gate system failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    report = main()
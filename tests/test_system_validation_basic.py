"""
Basic System Validation Tests
Lightweight testing framework that doesn't require external dependencies.
"""

import sys
import json
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class BasicSystemValidator:
    """
    Basic system validation without heavy dependencies.
    """
    
    def __init__(self):
        self.results = []
    
    def validate_module_imports(self) -> ValidationResult:
        """Test that all core modules can be imported."""
        import_results = {}
        total_modules = 0
        successful_imports = 0
        
        # Core modules to test
        modules_to_test = [
            'av_separation',
            'av_separation.config',
            'av_separation.research',
            'av_separation.robust', 
            'av_separation.optimized'
        ]
        
        for module_name in modules_to_test:
            total_modules += 1
            try:
                __import__(module_name)
                import_results[module_name] = True
                successful_imports += 1
            except ImportError as e:
                import_results[module_name] = f"ImportError: {e}"
            except Exception as e:
                import_results[module_name] = f"Error: {e}"
        
        score = successful_imports / total_modules if total_modules > 0 else 0
        
        return ValidationResult(
            test_name='module_imports',
            passed=score >= 0.8,  # 80% of modules should import
            score=score,
            details={
                'total_modules': total_modules,
                'successful_imports': successful_imports,
                'import_results': import_results
            }
        )
    
    def validate_file_structure(self) -> ValidationResult:
        """Validate that expected files and directories exist."""
        base_path = Path(__file__).parent.parent
        
        expected_structure = {
            'src/av_separation/__init__.py': 'Core module init',
            'src/av_separation/config.py': 'Configuration module',
            'src/av_separation/research/__init__.py': 'Research module init',
            'src/av_separation/research/novel_architectures.py': 'Novel architectures',
            'src/av_separation/research/experimental_benchmarks.py': 'Benchmarking',
            'src/av_separation/robust/__init__.py': 'Robust module init',
            'src/av_separation/robust/error_handling.py': 'Error handling',
            'src/av_separation/robust/validation.py': 'Input validation',
            'src/av_separation/robust/security_monitor.py': 'Security monitoring',
            'src/av_separation/optimized/__init__.py': 'Optimized module init',
            'src/av_separation/optimized/performance_engine.py': 'Performance engine',
            'src/av_separation/optimized/auto_scaler.py': 'Auto scaler',
            'src/av_separation/optimized/distributed_engine.py': 'Distributed engine',
            'README.md': 'Project README',
            'requirements.txt': 'Dependencies',
            'setup.py': 'Setup script'
        }
        
        file_results = {}
        total_files = len(expected_structure)
        existing_files = 0
        
        for file_path, description in expected_structure.items():
            full_path = base_path / file_path
            if full_path.exists():
                file_results[file_path] = f"EXISTS: {description}"
                existing_files += 1
            else:
                file_results[file_path] = f"MISSING: {description}"
        
        score = existing_files / total_files
        
        return ValidationResult(
            test_name='file_structure',
            passed=score >= 0.9,  # 90% of files should exist
            score=score,
            details={
                'total_files': total_files,
                'existing_files': existing_files,
                'file_results': file_results
            }
        )
    
    def validate_configuration_loading(self) -> ValidationResult:
        """Test configuration loading and validation."""
        try:
            # Try to import and create basic configuration
            from av_separation.config import SeparatorConfig
            
            # Test default configuration
            config = SeparatorConfig()
            
            # Basic validation
            has_model_config = hasattr(config, 'model')
            has_audio_config = hasattr(config, 'audio') 
            has_video_config = hasattr(config, 'video')
            
            config_checks = {
                'has_model_config': has_model_config,
                'has_audio_config': has_audio_config,
                'has_video_config': has_video_config
            }
            
            passed_checks = sum(config_checks.values())
            total_checks = len(config_checks)
            score = passed_checks / total_checks
            
            return ValidationResult(
                test_name='configuration_loading',
                passed=score >= 0.8,
                score=score,
                details={
                    'config_checks': config_checks,
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            )
        
        except Exception as e:
            return ValidationResult(
                test_name='configuration_loading',
                passed=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def validate_error_handling_module(self) -> ValidationResult:
        """Test error handling module functionality."""
        try:
            from av_separation.robust.error_handling import RobustErrorHandler
            
            # Test basic instantiation
            error_handler = RobustErrorHandler()
            
            # Test health metrics
            health_metrics = error_handler.get_health_metrics()
            
            # Basic checks
            checks = {
                'handler_created': error_handler is not None,
                'health_metrics_returned': isinstance(health_metrics, dict),
                'has_error_count': 'error_count_last_hour' in health_metrics,
                'has_memory_usage': 'memory_usage_ratio' in health_metrics
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            score = passed_checks / total_checks
            
            return ValidationResult(
                test_name='error_handling_module',
                passed=score >= 0.75,
                score=score,
                details={
                    'checks': checks,
                    'health_metrics_keys': list(health_metrics.keys()),
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            )
        
        except Exception as e:
            return ValidationResult(
                test_name='error_handling_module',
                passed=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def validate_validation_module(self) -> ValidationResult:
        """Test validation module functionality."""
        try:
            from av_separation.robust.validation import ComprehensiveValidator
            
            # Test basic instantiation
            validator = ComprehensiveValidator()
            
            # Test with mock data
            mock_config = {
                'model': {'d_model': 512, 'n_heads': 8, 'n_layers': 6},
                'audio': {'sample_rate': 16000, 'hop_length': 512, 'n_fft': 1024},
                'video': {'fps': 30, 'height': 224, 'width': 224}
            }
            
            validation_results = validator.validate_inputs(config=mock_config)
            
            checks = {
                'validator_created': validator is not None,
                'validation_results_returned': isinstance(validation_results, dict),
                'config_validated': 'config' in validation_results,
                'has_validation_summary': hasattr(validator, 'get_validation_summary')
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            score = passed_checks / total_checks
            
            return ValidationResult(
                test_name='validation_module',
                passed=score >= 0.75,
                score=score,
                details={
                    'checks': checks,
                    'validation_result_keys': list(validation_results.keys()),
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            )
        
        except Exception as e:
            return ValidationResult(
                test_name='validation_module',
                passed=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def validate_research_modules(self) -> ValidationResult:
        """Test research module imports and basic functionality."""
        try:
            # Test research module imports
            import_checks = {}
            
            modules_to_test = [
                'av_separation.research',
                'av_separation.research.novel_architectures',
                'av_separation.research.experimental_benchmarks'
            ]
            
            successful_imports = 0
            
            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    import_checks[module_name] = True
                    successful_imports += 1
                except Exception as e:
                    import_checks[module_name] = str(e)
            
            score = successful_imports / len(modules_to_test)
            
            return ValidationResult(
                test_name='research_modules',
                passed=score >= 0.6,  # Lower threshold as these may have dependencies
                score=score,
                details={
                    'import_checks': import_checks,
                    'successful_imports': successful_imports,
                    'total_modules': len(modules_to_test)
                }
            )
        
        except Exception as e:
            return ValidationResult(
                test_name='research_modules',
                passed=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def validate_optimized_modules(self) -> ValidationResult:
        """Test optimized module imports and basic functionality."""
        try:
            # Test optimized module imports
            import_checks = {}
            
            modules_to_test = [
                'av_separation.optimized',
                'av_separation.optimized.performance_engine',
                'av_separation.optimized.auto_scaler',
                'av_separation.optimized.distributed_engine'
            ]
            
            successful_imports = 0
            
            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    import_checks[module_name] = True
                    successful_imports += 1
                except Exception as e:
                    import_checks[module_name] = str(e)
            
            score = successful_imports / len(modules_to_test)
            
            return ValidationResult(
                test_name='optimized_modules',
                passed=score >= 0.6,  # Lower threshold as these may have dependencies
                score=score,
                details={
                    'import_checks': import_checks,
                    'successful_imports': successful_imports,
                    'total_modules': len(modules_to_test)
                }
            )
        
        except Exception as e:
            return ValidationResult(
                test_name='optimized_modules',
                passed=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests."""
        self.results = []
        
        # Run each validation test
        validations = [
            self.validate_module_imports,
            self.validate_file_structure,
            self.validate_configuration_loading,
            self.validate_error_handling_module,
            self.validate_validation_module,
            self.validate_research_modules,
            self.validate_optimized_modules
        ]
        
        for validation_func in validations:
            try:
                result = validation_func()
                self.results.append(result)
            except Exception as e:
                # Create error result for failed validation
                result = ValidationResult(
                    test_name=validation_func.__name__,
                    passed=False,
                    score=0.0,
                    details={},
                    error_message=str(e)
                )
                self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.results:
            self.run_all_validations()
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        average_score = sum(r.score for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Calculate weighted score (some tests are more important)
        weights = {
            'validate_module_imports': 1.5,
            'validate_file_structure': 1.2,
            'validate_configuration_loading': 1.3,
            'validate_error_handling_module': 1.1,
            'validate_validation_module': 1.1,
            'validate_research_modules': 0.8,  # Optional advanced features
            'validate_optimized_modules': 0.8   # Optional advanced features
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.results:
            weight = weights.get(result.test_name, 1.0)
            weighted_score += result.score * weight
            total_weight += weight
        
        final_weighted_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine overall status
        if final_weighted_score >= 0.85:
            overall_status = "EXCELLENT"
        elif final_weighted_score >= 0.75:
            overall_status = "GOOD"
        elif final_weighted_score >= 0.65:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        return {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_score': average_score,
            'weighted_score': final_weighted_score,
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'error_message': r.error_message,
                    'details': r.details
                }
                for r in self.results
            ]
        }


class TestSystemValidation(unittest.TestCase):
    """
    Unit test wrapper for system validation.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.validator = BasicSystemValidator()
    
    def test_module_imports(self):
        """Test module imports."""
        result = self.validator.validate_module_imports()
        self.assertTrue(result.passed, f"Module import test failed: {result.error_message}")
        self.assertGreaterEqual(result.score, 0.8)
    
    def test_file_structure(self):
        """Test file structure."""
        result = self.validator.validate_file_structure()
        self.assertTrue(result.passed, f"File structure test failed: {result.error_message}")
        self.assertGreaterEqual(result.score, 0.9)
    
    def test_configuration_loading(self):
        """Test configuration loading."""
        result = self.validator.validate_configuration_loading()
        self.assertTrue(result.passed, f"Configuration test failed: {result.error_message}")
        self.assertGreaterEqual(result.score, 0.8)
    
    def test_error_handling_module(self):
        """Test error handling module."""
        result = self.validator.validate_error_handling_module()
        self.assertTrue(result.passed, f"Error handling test failed: {result.error_message}")
        self.assertGreaterEqual(result.score, 0.75)
    
    def test_validation_module(self):
        """Test validation module."""
        result = self.validator.validate_validation_module()
        self.assertTrue(result.passed, f"Validation module test failed: {result.error_message}")
        self.assertGreaterEqual(result.score, 0.75)


def main():
    """Main function to run validation and generate report."""
    print("Starting AV-Separation System Validation...")
    
    validator = BasicSystemValidator()
    results = validator.run_all_validations()
    report = validator.generate_report()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SYSTEM VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    print(f"Weighted Score: {report['weighted_score']:.3f}")
    print(f"{'='*50}")
    
    # Print individual test results
    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} {result.test_name}: {result.score:.3f}")
        if result.error_message:
            print(f"    Error: {result.error_message}")
    
    # Save detailed report
    report_file = Path('system_validation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return success status
    return report['overall_status'] in ['EXCELLENT', 'GOOD', 'ACCEPTABLE']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
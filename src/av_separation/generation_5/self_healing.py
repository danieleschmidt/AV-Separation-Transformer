"""
GENERATION 5: SELF-HEALING SYSTEM
Autonomous bug detection, diagnosis, and resolution without human intervention
"""

import json
import time
import traceback
import hashlib
import subprocess
import ast
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import inspect
import warnings


@dataclass
class BugReport:
    """Comprehensive bug report"""
    bug_id: str
    timestamp: str
    severity: str  # critical, high, medium, low
    category: str  # runtime, logic, performance, memory, security
    title: str
    description: str
    stack_trace: Optional[str]
    affected_components: List[str]
    reproduction_steps: List[str]
    error_context: Dict[str, Any]
    suggested_fixes: List[str]
    confidence: float
    auto_fix_available: bool
    fix_complexity: str  # simple, moderate, complex


@dataclass
class FixAttempt:
    """Record of attempted fix"""
    fix_id: str
    bug_id: str
    timestamp: str
    fix_type: str
    fix_description: str
    code_changes: List[str]
    success: bool
    validation_results: Dict[str, Any]
    rollback_info: Optional[str]


class CodeAnalyzer:
    """Analyzes code for potential bugs and issues"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.known_patterns = self._load_bug_patterns()
        
    def analyze_code(self, code: str, filename: str = "unknown") -> List[Dict]:
        """Analyze code for potential bugs"""
        
        issues = []
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Run various analysis passes
            issues.extend(self._check_syntax_issues(tree, code))
            issues.extend(self._check_logic_issues(tree, code))
            issues.extend(self._check_performance_issues(tree, code))
            issues.extend(self._check_security_issues(tree, code))
            issues.extend(self._check_reliability_issues(tree, code))
            
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'severity': 'critical',
                'message': f"Syntax error: {e.msg}",
                'line': e.lineno,
                'column': e.offset,
                'fix_suggestion': self._suggest_syntax_fix(e)
            })
        
        except Exception as e:
            issues.append({
                'type': 'analysis_error',
                'severity': 'medium',
                'message': f"Code analysis failed: {str(e)}",
                'fix_suggestion': "Manual code review required"
            })
        
        return issues
    
    def _check_syntax_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        """Check for syntax-related issues"""
        issues = []
        
        class SyntaxChecker(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check for empty functions without docstrings
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Pass) and
                    not ast.get_docstring(node)):
                    issues.append({
                        'type': 'empty_function',
                        'severity': 'low',
                        'message': f"Empty function '{node.name}' without documentation",
                        'line': node.lineno,
                        'fix_suggestion': "Add docstring or implementation"
                    })
                
                # Check for unused parameters
                param_names = [arg.arg for arg in node.args.args]
                used_names = set()
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Name):
                        used_names.add(child.id)
                
                unused_params = [name for name in param_names if name not in used_names and name != 'self']
                if unused_params:
                    issues.append({
                        'type': 'unused_parameters',
                        'severity': 'low',
                        'message': f"Unused parameters in function '{node.name}': {unused_params}",
                        'line': node.lineno,
                        'fix_suggestion': f"Remove unused parameters or prefix with underscore: {['_' + p for p in unused_params]}"
                    })
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for unused imports (simplified)
                for alias in node.names:
                    issues.append({
                        'type': 'potential_unused_import',
                        'severity': 'low',
                        'message': f"Import '{alias.name}' may be unused",
                        'line': node.lineno,
                        'fix_suggestion': "Remove if unused or use import analysis tools"
                    })
                self.generic_visit(node)
        
        checker = SyntaxChecker()
        checker.visit(tree)
        
        return issues
    
    def _check_logic_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        """Check for logic-related issues"""
        issues = []
        
        class LogicChecker(ast.NodeVisitor):
            def visit_If(self, node):
                # Check for always true/false conditions
                if isinstance(node.test, ast.Constant):
                    if node.test.value:
                        issues.append({
                            'type': 'always_true_condition',
                            'severity': 'medium',
                            'message': "Condition is always True",
                            'line': node.lineno,
                            'fix_suggestion': "Remove condition or fix logic"
                        })
                    else:
                        issues.append({
                            'type': 'always_false_condition',
                            'severity': 'medium',
                            'message': "Condition is always False",
                            'line': node.lineno,
                            'fix_suggestion': "Remove branch or fix logic"
                        })
                
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                # Check for potential comparison issues
                if len(node.ops) > 1:
                    # Chained comparisons
                    for i, op in enumerate(node.ops):
                        if isinstance(op, ast.Is) or isinstance(op, ast.IsNot):
                            issues.append({
                                'type': 'identity_comparison',
                                'severity': 'medium',
                                'message': "Using 'is' in chained comparison may be unintended",
                                'line': node.lineno,
                                'fix_suggestion': "Consider using '==' for value comparison"
                            })
                
                self.generic_visit(node)
            
            def visit_For(self, node):
                # Check for potential infinite loops (simplified)
                if isinstance(node.iter, ast.Call):
                    if (isinstance(node.iter.func, ast.Name) and 
                        node.iter.func.id == 'range' and
                        len(node.iter.args) == 1):
                        issues.append({
                            'type': 'potential_performance_issue',
                            'severity': 'low',
                            'message': "Large range in loop may impact performance",
                            'line': node.lineno,
                            'fix_suggestion': "Consider using generators or chunking for large ranges"
                        })
                
                self.generic_visit(node)
        
        checker = LogicChecker()
        checker.visit(tree)
        
        return issues
    
    def _check_performance_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        """Check for performance-related issues"""
        issues = []
        
        class PerformanceChecker(ast.NodeVisitor):
            def visit_ListComp(self, node):
                # Check for nested list comprehensions
                for generator in node.generators:
                    if any(isinstance(child, ast.ListComp) for child in ast.walk(generator.iter)):
                        issues.append({
                            'type': 'nested_list_comprehension',
                            'severity': 'medium',
                            'message': "Nested list comprehensions may impact performance",
                            'line': node.lineno,
                            'fix_suggestion': "Consider using generator expressions or separate loops"
                        })
                
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                # Check for string concatenation in loops (simplified check)
                if isinstance(node.op, ast.Add):
                    if (isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str) or
                        isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) or
                        isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                        issues.append({
                            'type': 'string_concatenation',
                            'severity': 'low',
                            'message': "String concatenation may be inefficient in loops",
                            'line': node.lineno,
                            'fix_suggestion': "Consider using join() or f-strings for multiple concatenations"
                        })
                
                self.generic_visit(node)
        
        checker = PerformanceChecker()
        checker.visit(tree)
        
        return issues
    
    def _check_security_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        """Check for security-related issues"""
        issues = []
        
        class SecurityChecker(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for dangerous function calls
                dangerous_functions = ['eval', 'exec', 'compile']
                
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                    issues.append({
                        'type': 'dangerous_function',
                        'severity': 'high',
                        'message': f"Use of potentially dangerous function: {node.func.id}",
                        'line': node.lineno,
                        'fix_suggestion': f"Avoid {node.func.id} or validate input thoroughly"
                    })
                
                # Check for shell command execution
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id in ['os', 'subprocess'] and
                    node.func.attr in ['system', 'popen', 'call']):
                    issues.append({
                        'type': 'shell_execution',
                        'severity': 'high',
                        'message': "Shell command execution detected",
                        'line': node.lineno,
                        'fix_suggestion': "Validate and sanitize all inputs to shell commands"
                    })
                
                self.generic_visit(node)
        
        checker = SecurityChecker()
        checker.visit(tree)
        
        return issues
    
    def _check_reliability_issues(self, tree: ast.AST, code: str) -> List[Dict]:
        """Check for reliability-related issues"""
        issues = []
        
        class ReliabilityChecker(ast.NodeVisitor):
            def visit_Try(self, node):
                # Check for bare except clauses
                for handler in node.handlers:
                    if handler.type is None:
                        issues.append({
                            'type': 'bare_except',
                            'severity': 'medium',
                            'message': "Bare except clause catches all exceptions",
                            'line': handler.lineno,
                            'fix_suggestion': "Specify exception types or use 'except Exception:'"
                        })
                
                # Check for empty except blocks
                for handler in node.handlers:
                    if (len(handler.body) == 1 and 
                        isinstance(handler.body[0], ast.Pass)):
                        issues.append({
                            'type': 'empty_except',
                            'severity': 'medium',
                            'message': "Empty except block silently ignores errors",
                            'line': handler.lineno,
                            'fix_suggestion': "Add proper error handling or logging"
                        })
                
                self.generic_visit(node)
            
            def visit_Assert(self, node):
                # Check for assert statements in production code
                issues.append({
                    'type': 'assert_statement',
                    'severity': 'low',
                    'message': "Assert statements are disabled with -O flag",
                    'line': node.lineno,
                    'fix_suggestion': "Use proper error handling instead of assert for production code"
                })
                
                self.generic_visit(node)
        
        checker = ReliabilityChecker()
        checker.visit(tree)
        
        return issues
    
    def _suggest_syntax_fix(self, syntax_error: SyntaxError) -> str:
        """Suggest fix for syntax errors"""
        
        error_msg = syntax_error.msg.lower()
        
        if "invalid syntax" in error_msg:
            return "Check for missing colons, parentheses, or quotes"
        elif "unexpected eof" in error_msg:
            return "Check for unclosed brackets, parentheses, or quotes"
        elif "invalid character" in error_msg:
            return "Check for invalid characters or encoding issues"
        else:
            return "Review syntax near the error location"
    
    def _load_bug_patterns(self) -> Dict:
        """Load known bug patterns"""
        return {
            'null_pointer': {
                'pattern': r'\.(\w+)\s*=\s*None.*\1\.',
                'message': 'Potential null pointer access',
                'fix': 'Add null check before access'
            },
            'resource_leak': {
                'pattern': r'open\([^)]+\)(?!.*with)',
                'message': 'Potential resource leak',
                'fix': 'Use context manager (with statement)'
            },
            'division_by_zero': {
                'pattern': r'/\s*[a-zA-Z_]\w*(?!\s*[!=]=)',
                'message': 'Potential division by zero',
                'fix': 'Add zero check before division'
            }
        }


class AutoFixEngine:
    """Automatically generates and applies fixes for detected bugs"""
    
    def __init__(self):
        self.fix_templates = self._load_fix_templates()
        self.fix_history = []
        
    def generate_fix(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Generate automatic fix for bug"""
        
        category = bug_report.category
        severity = bug_report.severity
        
        # Only attempt auto-fix for simple, well-understood issues
        if severity in ['critical', 'high'] or bug_report.fix_complexity == 'complex':
            return None
        
        fix_method = getattr(self, f'_fix_{category}', None)
        if fix_method:
            return fix_method(bug_report, code)
        
        return None
    
    def _fix_syntax(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Fix syntax-related issues"""
        
        lines = code.split('\n')
        
        if 'empty_function' in bug_report.description:
            # Add basic docstring
            function_line = None
            for i, line in enumerate(lines):
                if 'def ' in line and ':' in line:
                    function_line = i
                    break
            
            if function_line is not None:
                indent = len(lines[function_line]) - len(lines[function_line].lstrip())
                docstring = ' ' * (indent + 4) + '"""TODO: Implement this function."""'
                lines.insert(function_line + 1, docstring)
                return '\n'.join(lines)
        
        elif 'unused_parameters' in bug_report.description:
            # Prefix unused parameters with underscore
            # This is a simplified implementation
            for i, line in enumerate(lines):
                if 'def ' in line:
                    # Extract and modify parameter list
                    # Simplified regex-based approach
                    import re
                    pattern = r'def\s+\w+\s*\(([^)]*)\)'
                    match = re.search(pattern, line)
                    if match:
                        params = match.group(1)
                        # This would need more sophisticated parsing in reality
                        lines[i] = line  # Placeholder - would implement parameter renaming
                
        return None
    
    def _fix_logic(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Fix logic-related issues"""
        
        lines = code.split('\n')
        
        if 'always_true_condition' in bug_report.description:
            # Comment out the condition
            for i, line in enumerate(lines):
                if 'if True:' in line:
                    lines[i] = line.replace('if True:', '# if True:  # TODO: Fix condition')
                    break
            return '\n'.join(lines)
        
        elif 'always_false_condition' in bug_report.description:
            # Comment out the entire block
            for i, line in enumerate(lines):
                if 'if False:' in line:
                    lines[i] = line.replace('if False:', '# if False:  # TODO: Fix or remove')
                    # Would need to comment out the entire block
                    break
            return '\n'.join(lines)
        
        return None
    
    def _fix_performance(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Fix performance-related issues"""
        
        if 'string_concatenation' in bug_report.description:
            # Suggest using join instead of concatenation
            # This would require more sophisticated code transformation
            # For now, add a comment
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if '+' in line and '"' in line:
                    lines[i] = line + '  # TODO: Consider using join() for better performance'
                    break
            return '\n'.join(lines)
        
        return None
    
    def _fix_security(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Fix security-related issues"""
        
        lines = code.split('\n')
        
        if 'dangerous_function' in bug_report.description:
            # Add warning comment
            for i, line in enumerate(lines):
                if 'eval(' in line or 'exec(' in line:
                    lines.insert(i, '    # WARNING: Security risk - validate input thoroughly')
                    break
            return '\n'.join(lines)
        
        return None
    
    def _fix_reliability(self, bug_report: BugReport, code: str) -> Optional[str]:
        """Fix reliability-related issues"""
        
        lines = code.split('\n')
        
        if 'bare_except' in bug_report.description:
            # Replace bare except with Exception
            for i, line in enumerate(lines):
                if 'except:' in line:
                    lines[i] = line.replace('except:', 'except Exception as e:')
                    # Add logging line
                    indent = len(line) - len(line.lstrip())
                    log_line = ' ' * (indent + 4) + 'print(f"Error occurred: {e}")  # TODO: Use proper logging'
                    lines.insert(i + 1, log_line)
                    break
            return '\n'.join(lines)
        
        elif 'empty_except' in bug_report.description:
            # Add basic error handling
            for i, line in enumerate(lines):
                if 'except' in line and ':' in line:
                    # Find the pass statement
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if 'pass' in lines[j]:
                            indent = len(lines[j]) - len(lines[j].lstrip())
                            lines[j] = ' ' * indent + 'print("Error occurred but was ignored")  # TODO: Add proper error handling'
                            break
                    break
            return '\n'.join(lines)
        
        return None
    
    def _load_fix_templates(self) -> Dict:
        """Load fix templates for common issues"""
        
        return {
            'null_check': '''
if {variable} is not None:
    {original_code}
''',
            'exception_handling': '''
try:
    {original_code}
except Exception as e:
    print(f"Error: {e}")  # TODO: Add proper error handling
''',
            'resource_management': '''
with {resource_acquisition} as {resource_variable}:
    {original_code}
'''
        }


class ValidationEngine:
    """Validates fixes before and after application"""
    
    def __init__(self):
        self.test_cache = {}
        
    def validate_fix(self, original_code: str, fixed_code: str, bug_report: BugReport) -> Dict[str, Any]:
        """Validate that fix is safe and effective"""
        
        validation_results = {
            'syntax_valid': False,
            'logic_preserved': False,
            'bug_fixed': False,
            'no_new_issues': False,
            'performance_impact': 'unknown',
            'confidence': 0.0,
            'issues': []
        }
        
        # Syntax validation
        try:
            ast.parse(fixed_code)
            validation_results['syntax_valid'] = True
        except SyntaxError as e:
            validation_results['issues'].append(f"Syntax error in fix: {e}")
        
        # Logic preservation check (simplified)
        validation_results['logic_preserved'] = self._check_logic_preservation(original_code, fixed_code)
        
        # Bug fix validation
        validation_results['bug_fixed'] = self._check_bug_resolution(fixed_code, bug_report)
        
        # New issues check
        analyzer = CodeAnalyzer()
        new_issues = analyzer.analyze_code(fixed_code)
        old_issues = analyzer.analyze_code(original_code)
        
        if len(new_issues) <= len(old_issues):
            validation_results['no_new_issues'] = True
        else:
            validation_results['issues'].append("Fix introduced new issues")
        
        # Calculate confidence
        confidence_factors = [
            validation_results['syntax_valid'],
            validation_results['logic_preserved'],
            validation_results['bug_fixed'],
            validation_results['no_new_issues']
        ]
        validation_results['confidence'] = sum(confidence_factors) / len(confidence_factors)
        
        return validation_results
    
    def _check_logic_preservation(self, original: str, fixed: str) -> bool:
        """Check if core logic is preserved (simplified)"""
        
        # Basic heuristic: similar structure and length
        orig_lines = [line.strip() for line in original.split('\n') if line.strip()]
        fixed_lines = [line.strip() for line in fixed.split('\n') if line.strip()]
        
        # Allow for minor differences (comments, small additions)
        length_ratio = len(fixed_lines) / max(len(orig_lines), 1)
        
        return 0.8 <= length_ratio <= 1.5
    
    def _check_bug_resolution(self, fixed_code: str, bug_report: BugReport) -> bool:
        """Check if the bug has been resolved"""
        
        # Simple pattern matching for known bug types
        bug_indicators = {
            'bare_except': r'except\s*:',
            'empty_except': r'except.*:\s*pass',
            'unused_import': r'import\s+\w+.*#.*unused',
            'always_true': r'if\s+True\s*:',
            'always_false': r'if\s+False\s*:'
        }
        
        import re
        for bug_type, pattern in bug_indicators.items():
            if bug_type in bug_report.description.lower():
                if not re.search(pattern, fixed_code):
                    return True
        
        return False


class SelfHealingSystem:
    """Main self-healing system that coordinates bug detection and fixing"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.code_analyzer = CodeAnalyzer()
        self.auto_fix_engine = AutoFixEngine()
        self.validation_engine = ValidationEngine()
        
        self.bug_reports = []
        self.fix_attempts = []
        self.healing_stats = {
            'bugs_detected': 0,
            'bugs_fixed': 0,
            'fix_success_rate': 0.0,
            'last_healing_cycle': None
        }
        
        # Configuration
        self.auto_fix_enabled = self.config.get('auto_fix_enabled', True)
        self.fix_confidence_threshold = self.config.get('fix_confidence_threshold', 0.8)
        self.backup_enabled = self.config.get('backup_enabled', True)
        
    def heal_code(self, code: str, filename: str = "unknown") -> Dict[str, Any]:
        """Main healing function - detect and fix bugs in code"""
        
        print(f"🏥 Self-healing: analyzing {filename}...")
        
        healing_report = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'original_code': code,
            'bugs_detected': 0,
            'bugs_fixed': 0,
            'fixes_applied': [],
            'validation_results': [],
            'final_code': code,
            'healing_success': False
        }
        
        try:
            # Step 1: Detect bugs
            issues = self.code_analyzer.analyze_code(code, filename)
            bug_reports = self._create_bug_reports(issues, code, filename)
            
            healing_report['bugs_detected'] = len(bug_reports)
            self.bug_reports.extend(bug_reports)
            
            if not bug_reports:
                print("✅ No bugs detected - code is healthy")
                healing_report['healing_success'] = True
                return healing_report
            
            print(f"🐛 Detected {len(bug_reports)} potential issues")
            
            # Step 2: Attempt fixes
            current_code = code
            fixes_applied = 0
            
            for bug_report in bug_reports:
                if not self.auto_fix_enabled:
                    continue
                
                print(f"🔧 Attempting fix for: {bug_report.title}")
                
                # Generate fix
                fixed_code = self.auto_fix_engine.generate_fix(bug_report, current_code)
                
                if not fixed_code:
                    print(f"   ⚠️  No automatic fix available")
                    continue
                
                # Validate fix
                validation = self.validation_engine.validate_fix(current_code, fixed_code, bug_report)
                
                if validation['confidence'] >= self.fix_confidence_threshold:
                    # Apply fix
                    if self.backup_enabled:
                        self._backup_code(current_code, filename)
                    
                    current_code = fixed_code
                    fixes_applied += 1
                    
                    fix_attempt = FixAttempt(
                        fix_id=f"fix_{int(time.time())}_{fixes_applied}",
                        bug_id=bug_report.bug_id,
                        timestamp=datetime.now().isoformat(),
                        fix_type=bug_report.category,
                        fix_description=f"Auto-fix for {bug_report.title}",
                        code_changes=[f"Applied fix to {filename}"],
                        success=True,
                        validation_results=validation,
                        rollback_info=f"backup_{filename}_{int(time.time())}"
                    )
                    
                    self.fix_attempts.append(fix_attempt)
                    healing_report['fixes_applied'].append(asdict(fix_attempt))
                    
                    print(f"   ✅ Fix applied (confidence: {validation['confidence']:.2f})")
                    
                else:
                    print(f"   ❌ Fix rejected (confidence: {validation['confidence']:.2f} < {self.fix_confidence_threshold})")
            
            healing_report['bugs_fixed'] = fixes_applied
            healing_report['final_code'] = current_code
            healing_report['healing_success'] = fixes_applied > 0
            
            # Update statistics
            self._update_healing_stats(len(bug_reports), fixes_applied)
            
            if fixes_applied > 0:
                print(f"🎉 Healing complete: {fixes_applied}/{len(bug_reports)} issues fixed")
            else:
                print("⚠️  No automatic fixes could be safely applied")
            
        except Exception as e:
            print(f"❌ Self-healing failed: {str(e)}")
            healing_report['error'] = str(e)
            healing_report['stack_trace'] = traceback.format_exc()
        
        return healing_report
    
    def monitor_and_heal(self, code_paths: List[str]) -> Dict[str, Any]:
        """Monitor multiple code files and heal them"""
        
        print("🔍 Starting continuous code monitoring and healing...")
        
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': 0,
            'total_bugs_detected': 0,
            'total_bugs_fixed': 0,
            'file_reports': [],
            'overall_health_improvement': 0.0
        }
        
        for code_path in code_paths:
            try:
                with open(code_path, 'r') as f:
                    code = f.read()
                
                healing_report = self.heal_code(code, code_path)
                
                monitoring_report['files_processed'] += 1
                monitoring_report['total_bugs_detected'] += healing_report['bugs_detected']
                monitoring_report['total_bugs_fixed'] += healing_report['bugs_fixed']
                monitoring_report['file_reports'].append(healing_report)
                
                # Save healed code if fixes were applied
                if healing_report['healing_success'] and healing_report['final_code'] != code:
                    with open(code_path, 'w') as f:
                        f.write(healing_report['final_code'])
                    print(f"💾 Saved healed code to {code_path}")
                
            except Exception as e:
                print(f"❌ Failed to process {code_path}: {str(e)}")
                monitoring_report['file_reports'].append({
                    'filename': code_path,
                    'error': str(e),
                    'healing_success': False
                })
        
        # Calculate overall health improvement
        if monitoring_report['total_bugs_detected'] > 0:
            improvement = monitoring_report['total_bugs_fixed'] / monitoring_report['total_bugs_detected']
            monitoring_report['overall_health_improvement'] = improvement
        
        return monitoring_report
    
    def _create_bug_reports(self, issues: List[Dict], code: str, filename: str) -> List[BugReport]:
        """Create comprehensive bug reports from detected issues"""
        
        bug_reports = []
        
        for issue in issues:
            bug_id = hashlib.md5(f"{filename}_{issue}_{int(time.time())}".encode()).hexdigest()[:8]
            
            # Classify fix complexity
            complexity = self._classify_fix_complexity(issue)
            
            # Determine if auto-fix is available
            auto_fix_available = (complexity == 'simple' and 
                                issue.get('severity', 'unknown') in ['low', 'medium'])
            
            bug_report = BugReport(
                bug_id=bug_id,
                timestamp=datetime.now().isoformat(),
                severity=issue.get('severity', 'medium'),
                category=issue.get('type', 'unknown'),
                title=issue.get('message', 'Unknown issue'),
                description=f"{issue.get('message', 'Unknown issue')} in {filename}",
                stack_trace=None,
                affected_components=[filename],
                reproduction_steps=[f"Analyze code in {filename} at line {issue.get('line', 0)}"],
                error_context=issue,
                suggested_fixes=[issue.get('fix_suggestion', 'Manual review required')],
                confidence=0.8,  # Based on static analysis
                auto_fix_available=auto_fix_available,
                fix_complexity=complexity
            )
            
            bug_reports.append(bug_report)
        
        return bug_reports
    
    def _classify_fix_complexity(self, issue: Dict) -> str:
        """Classify the complexity of fixing an issue"""
        
        simple_fixes = [
            'empty_function', 'unused_parameters', 'bare_except', 
            'empty_except', 'string_concatenation'
        ]
        
        moderate_fixes = [
            'always_true_condition', 'always_false_condition',
            'dangerous_function', 'assert_statement'
        ]
        
        issue_type = issue.get('type', 'unknown')
        
        if issue_type in simple_fixes:
            return 'simple'
        elif issue_type in moderate_fixes:
            return 'moderate'
        else:
            return 'complex'
    
    def _backup_code(self, code: str, filename: str):
        """Create backup of original code"""
        
        backup_path = Path(f"{filename}.backup_{int(time.time())}")
        with open(backup_path, 'w') as f:
            f.write(code)
        
        print(f"💾 Backup created: {backup_path}")
    
    def _update_healing_stats(self, bugs_detected: int, bugs_fixed: int):
        """Update healing statistics"""
        
        self.healing_stats['bugs_detected'] += bugs_detected
        self.healing_stats['bugs_fixed'] += bugs_fixed
        
        if self.healing_stats['bugs_detected'] > 0:
            self.healing_stats['fix_success_rate'] = (
                self.healing_stats['bugs_fixed'] / self.healing_stats['bugs_detected']
            )
        
        self.healing_stats['last_healing_cycle'] = datetime.now().isoformat()
    
    def get_healing_summary(self) -> Dict[str, Any]:
        """Get comprehensive healing summary"""
        
        summary = {
            'healing_statistics': self.healing_stats.copy(),
            'recent_bug_reports': [asdict(bug) for bug in self.bug_reports[-10:]],
            'recent_fix_attempts': [asdict(fix) for fix in self.fix_attempts[-10:]],
            'system_health': self._assess_system_health(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _assess_system_health(self) -> str:
        """Assess overall system health"""
        
        if not self.healing_stats['bugs_detected']:
            return 'excellent'
        
        success_rate = self.healing_stats['fix_success_rate']
        
        if success_rate >= 0.9:
            return 'excellent'
        elif success_rate >= 0.7:
            return 'good'
        elif success_rate >= 0.5:
            return 'fair'
        else:
            return 'needs_attention'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement"""
        
        recommendations = []
        
        if self.healing_stats['fix_success_rate'] < 0.7:
            recommendations.append("Consider manual review of complex issues")
        
        if len(self.bug_reports) > 50:
            recommendations.append("High number of issues detected - consider code quality review")
        
        if not self.auto_fix_enabled:
            recommendations.append("Enable auto-fix for faster issue resolution")
        
        return recommendations
    
    def export_healing_data(self, filepath: str):
        """Export healing data to file"""
        
        export_data = {
            'healing_summary': self.get_healing_summary(),
            'all_bug_reports': [asdict(bug) for bug in self.bug_reports],
            'all_fix_attempts': [asdict(fix) for fix in self.fix_attempts],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Healing data exported to {filepath}")


# Demonstration function
def demonstrate_self_healing():
    """Demonstrate self-healing capabilities"""
    
    print("🏥 GENERATION 5: SELF-HEALING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize self-healing system
    healing_system = SelfHealingSystem({
        'auto_fix_enabled': True,
        'fix_confidence_threshold': 0.7,
        'backup_enabled': True
    })
    
    # Example buggy code
    buggy_code = '''
def process_data(data, threshold):
    results = []
    
    # Always true condition
    if True:
        print("Processing data...")
    
    # Bare except clause
    try:
        for item in data:
            if item / threshold > 1.0:  # Potential division by zero
                results.append(item)
    except:
        pass  # Empty except block
    
    # Unused parameter
    def helper_function(x, unused_param):
        return x * 2
    
    return results

# Unused import
import os
'''
    
    print("🐛 Analyzing buggy code sample...")
    print("Code issues:")
    print("- Always true condition")
    print("- Bare except clause")
    print("- Empty except block")
    print("- Potential division by zero")
    print("- Unused parameter")
    print("- Unused import")
    print()
    
    # Run healing process
    healing_report = healing_system.heal_code(buggy_code, "example.py")
    
    print("\n📋 HEALING REPORT")
    print("-" * 40)
    print(f"Bugs Detected: {healing_report['bugs_detected']}")
    print(f"Bugs Fixed: {healing_report['bugs_fixed']}")
    print(f"Healing Success: {healing_report['healing_success']}")
    
    if healing_report['fixes_applied']:
        print("\nFixes Applied:")
        for fix in healing_report['fixes_applied']:
            print(f"  • {fix['fix_description']}")
    
    # Show healed code if different
    if healing_report['final_code'] != buggy_code:
        print("\n🔧 HEALED CODE PREVIEW:")
        print("-" * 40)
        healed_lines = healing_report['final_code'].split('\n')
        for i, line in enumerate(healed_lines[:20], 1):  # Show first 20 lines
            print(f"{i:2d}: {line}")
        if len(healed_lines) > 20:
            print("    ... (truncated)")
    
    # Show healing summary
    summary = healing_system.get_healing_summary()
    
    print("\n📊 SYSTEM HEALTH SUMMARY")
    print("-" * 40)
    stats = summary['healing_statistics']
    print(f"Total Bugs Detected: {stats['bugs_detected']}")
    print(f"Total Bugs Fixed: {stats['bugs_fixed']}")
    print(f"Fix Success Rate: {stats['fix_success_rate']:.1%}")
    print(f"System Health: {summary['system_health'].upper()}")
    
    if summary['recommendations']:
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    # Export healing data
    healing_system.export_healing_data('self_healing_data.json')
    
    print("\n✅ Self-healing demonstration complete!")
    
    return healing_system


if __name__ == "__main__":
    # Run demonstration
    demonstrate_self_healing()
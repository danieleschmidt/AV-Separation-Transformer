#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from av_separation import SeparatorConfig
from av_separation.models import AVSeparationTransformer
from av_separation.robust_core import (
    robust_error_handler, robust_context, RobustLogger,
    ErrorCategory, ErrorSeverity, CircuitBreaker, RetryHandler,
    global_health_checker
)
from av_separation.enhanced_security import (
    AdvancedInputValidator, SecurityAuditor, RateLimiter,
    SecurityEvent, ThreatLevel, global_validator
)

@robust_error_handler('demo', ErrorCategory.PROCESSING, ErrorSeverity.HIGH)
def robust_separation_test():
    """Test separation with robust error handling"""
    logger = RobustLogger('demo')
    
    with robust_context('demo', 'robust_separation'):
        # Create config with validation
        config = SeparatorConfig()
        logger.log_info("Configuration created", {'max_speakers': config.model.max_speakers})
        
        # Initialize security components
        validator = AdvancedInputValidator()
        auditor = SecurityAuditor()
        rate_limiter = RateLimiter(max_requests=10, time_window=60)
        
        # Test input validation
        test_inputs = [
            "normal_audio_file.wav",
            "<script>alert('xss')</script>",
            "audio_file_with_very_long_name" + "x" * 300,
            "../../../etc/passwd"
        ]
        
        for test_input in test_inputs:
            is_valid_filename, errors = validator.validate_filename(test_input)
            if not is_valid_filename:
                trace_id = auditor.log_security_event(
                    SecurityEvent.INVALID_INPUT,
                    ThreatLevel.MEDIUM,
                    f"Invalid filename: {errors}",
                    source_ip="127.0.0.1"
                )
                logger.log_warning(f"Security validation failed", {
                    'input': test_input[:50],
                    'errors': errors,
                    'trace_id': trace_id
                })
        
        # Test rate limiting
        client_ip = "test_client_127.0.0.1"
        for i in range(15):  # Exceed rate limit
            allowed, info = rate_limiter.is_allowed(client_ip)
            if not allowed:
                auditor.log_security_event(
                    SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    f"Rate limit exceeded: {info}",
                    source_ip=client_ip
                )
                logger.log_warning("Rate limit exceeded", info)
                break
        
        # Create model with circuit breaker protection
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
        
        def create_model_with_protection():
            model = AVSeparationTransformer(config)
            logger.log_info("Model created", {'parameters': model.get_num_params()})
            return model
        
        try:
            model = circuit_breaker.call(create_model_with_protection)
        except Exception as e:
            logger.log_warning("Circuit breaker triggered", {'error': str(e)})
            raise
        
        # Test model with retry mechanism
        retry_handler = RetryHandler(max_retries=2, base_delay=0.5)
        
        def test_inference():
            batch_size = 1
            seq_len = 50
            
            dummy_audio = torch.randn(batch_size, config.audio.n_mels, seq_len)
            dummy_video = torch.randn(batch_size, 15, 3, *config.video.image_size)
            
            model.eval()
            with torch.no_grad():
                outputs = model(dummy_audio, dummy_video)
                
            return {
                'waveform_shape': outputs['separated_waveforms'].shape,
                'speaker_logits_shape': outputs['speaker_logits'].shape,
                'alignment_score': outputs['alignment_score'].mean().item()
            }
        
        try:
            result = retry_handler.retry_sync(test_inference)
            logger.log_info("Inference successful", result)
        except Exception as e:
            logger.log_warning("Inference failed after retries", {'error': str(e)})
            raise
        
        # Test security summary
        security_summary = auditor.get_security_summary()
        logger.log_info("Security summary", security_summary)
        
        return True

async def test_health_checks():
    """Test system health monitoring"""
    logger = RobustLogger('health_check')
    
    # Run health checks
    health_results = await global_health_checker.run_checks()
    logger.log_info("Health check results", health_results)
    
    return health_results

def test_advanced_validation():
    """Test advanced input validation"""
    logger = RobustLogger('validation')
    validator = global_validator
    
    # Test malicious inputs
    malicious_inputs = [
        "SELECT * FROM users WHERE password='",
        "<script>document.location='http://evil.com'</script>",
        "javascript:alert('xss')",
        "eval('malicious code')",
        "../../etc/shadow",
        "cmd.exe /c format c:"
    ]
    
    validation_results = []
    
    for malicious_input in malicious_inputs:
        is_valid, errors = validator.validate_text_input(malicious_input, max_length=500)
        validation_results.append({
            'input': malicious_input[:30] + "..." if len(malicious_input) > 30 else malicious_input,
            'valid': is_valid,
            'errors': errors
        })
        
        if not is_valid:
            logger.log_warning("Malicious input detected", {
                'input_preview': malicious_input[:50],
                'errors': errors
            })
    
    return validation_results

def main():
    print("=== AV-Separation Transformer Robust System Demo ===")
    
    try:
        # Test robust separation
        print("\n1. Testing Robust Error Handling...")
        result = robust_separation_test()
        print(f"✓ Robust separation test: {'PASSED' if result else 'FAILED'}")
        
        # Test health monitoring
        print("\n2. Testing Health Monitoring...")
        import asyncio
        health_results = asyncio.run(test_health_checks())
        print(f"✓ Health check: {health_results['overall_status']}")
        
        # Test advanced validation
        print("\n3. Testing Advanced Security Validation...")
        validation_results = test_advanced_validation()
        blocked_inputs = sum(1 for result in validation_results if not result['valid'])
        print(f"✓ Security validation: Blocked {blocked_inputs}/{len(validation_results)} malicious inputs")
        
        print(f"\n=== Generation 2 (MAKE IT ROBUST) Complete ===")
        print("✅ Enhanced error handling, security, logging, and monitoring active")
        
    except Exception as e:
        print(f"✗ Robust system test: FAILED - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
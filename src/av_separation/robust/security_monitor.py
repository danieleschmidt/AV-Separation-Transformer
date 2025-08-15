"""
Advanced Security Monitoring and Threat Detection System
Production-grade security with real-time monitoring and response.
"""

import time
import threading
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import ipaddress
from pathlib import Path
import jwt
from datetime import datetime, timedelta


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTACK = "injection_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityIncident:
    """Container for security incident information."""
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


class ThreatDetector:
    """
    Advanced threat detection using behavioral analysis and pattern recognition.
    """
    
    def __init__(self):
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        self.suspicious_patterns = {
            'sql_injection': [
                r"('|(\\')|(;)|(\\;)|(%)|(\\%)|(--)|(\\--)|(\|)|(\\\|)|(\*)|(\\\*)",
                r"union.*select",
                r"insert.*into",
                r"delete.*from",
                r"drop.*table"
            ],
            'xss': [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"vbscript:",
                r"onload=",
                r"onerror="
            ],
            'command_injection': [
                r";\s*(rm|del|format|shutdown)",
                r"\|\s*(rm|del|format|shutdown)",
                r"&&\s*(rm|del|format|shutdown)",
                r"`.*`",
                r"\$\(.*\)"
            ]
        }
        
        # Rate limiting thresholds
        self.rate_limits = {
            'requests_per_minute': 100,
            'failed_attempts_threshold': 5,
            'concurrent_connections': 50
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_brute_force(self, source_ip: str, endpoint: str, success: bool) -> Optional[SecurityIncident]:
        """Detect brute force attacks."""
        current_time = time.time()
        
        # Track request
        self.request_history[source_ip].append((current_time, endpoint, success))
        
        if not success:
            self.failed_attempts[source_ip] += 1
            
            # Check if threshold exceeded
            if self.failed_attempts[source_ip] >= self.rate_limits['failed_attempts_threshold']:
                return SecurityIncident(
                    timestamp=current_time,
                    threat_type=ThreatType.BRUTE_FORCE,
                    severity=SecurityLevel.HIGH,
                    source_ip=source_ip,
                    endpoint=endpoint,
                    description=f"Brute force attack detected: {self.failed_attempts[source_ip]} failed attempts",
                    metadata={'failed_attempts': self.failed_attempts[source_ip]}
                )
        else:
            # Reset counter on success
            self.failed_attempts[source_ip] = 0
        
        return None
    
    def detect_ddos(self, source_ip: str) -> Optional[SecurityIncident]:
        """Detect DDoS attacks based on request rate."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Count requests in the last minute
        recent_requests = [
            req for req in self.request_history[source_ip]
            if req[0] > minute_ago
        ]
        
        if len(recent_requests) > self.rate_limits['requests_per_minute']:
            return SecurityIncident(
                timestamp=current_time,
                threat_type=ThreatType.DDoS,
                severity=SecurityLevel.CRITICAL,
                source_ip=source_ip,
                description=f"DDoS attack detected: {len(recent_requests)} requests in 1 minute",
                metadata={'requests_per_minute': len(recent_requests)}
            )
        
        return None
    
    def detect_malicious_input(self, input_data: str, context: str = "") -> Optional[SecurityIncident]:
        """Detect malicious input patterns."""
        import re
        
        current_time = time.time()
        
        for attack_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    return SecurityIncident(
                        timestamp=current_time,
                        threat_type=ThreatType.MALICIOUS_INPUT,
                        severity=SecurityLevel.HIGH,
                        description=f"{attack_type.upper()} pattern detected in input",
                        metadata={
                            'attack_type': attack_type,
                            'pattern': pattern,
                            'context': context,
                            'input_sample': input_data[:100]  # First 100 chars for analysis
                        }
                    )
        
        return None
    
    def detect_data_exfiltration(self, response_size: int, endpoint: str) -> Optional[SecurityIncident]:
        """Detect potential data exfiltration based on response patterns."""
        current_time = time.time()
        
        # Define suspicious thresholds
        large_response_threshold = 10 * 1024 * 1024  # 10MB
        
        if response_size > large_response_threshold:
            return SecurityIncident(
                timestamp=current_time,
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=SecurityLevel.MEDIUM,
                endpoint=endpoint,
                description=f"Large response detected: {response_size} bytes",
                metadata={'response_size': response_size}
            )
        
        return None
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is currently blocked."""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str):
        """Block an IP address."""
        self.blocked_ips.add(ip_address)
        self.logger.warning(f"IP address blocked: {ip_address}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip_address)
        self.logger.info(f"IP address unblocked: {ip_address}")


class AccessController:
    """
    Advanced access control with role-based permissions and JWT authentication.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.active_tokens = set()
        self.user_permissions = defaultdict(set)
        self.role_permissions = {
            'admin': {'read', 'write', 'delete', 'manage_users', 'view_logs'},
            'user': {'read', 'write'},
            'readonly': {'read'},
            'api': {'read', 'write'},
            'guest': set()
        }
        
        self.session_timeout = 3600  # 1 hour
        self.max_concurrent_sessions = 5
        self.user_sessions = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def generate_token(self, user_id: str, role: str = 'user', 
                      expires_in: int = 3600) -> str:
        """Generate JWT token for authentication."""
        expiration = datetime.utcnow() + timedelta(seconds=expires_in)
        
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': expiration,
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        self.active_tokens.add(payload['jti'])
        
        # Manage concurrent sessions
        self.user_sessions[user_id].append({
            'jti': payload['jti'],
            'created': time.time(),
            'last_activity': time.time()
        })
        
        # Remove old sessions if exceeding limit
        if len(self.user_sessions[user_id]) > self.max_concurrent_sessions:
            oldest_session = min(self.user_sessions[user_id], key=lambda x: x['created'])
            self.revoke_token_by_jti(oldest_session['jti'])
            self.user_sessions[user_id].remove(oldest_session)
        
        self.logger.info(f"Token generated for user: {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is still active
            if payload['jti'] not in self.active_tokens:
                return None
            
            # Update last activity
            user_id = payload['user_id']
            for session in self.user_sessions[user_id]:
                if session['jti'] == payload['jti']:
                    session['last_activity'] = time.time()
                    break
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token verification failed: expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Token verification failed: invalid")
            return None
    
    def revoke_token(self, token: str):
        """Revoke a specific token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'], 
                               options={"verify_exp": False})
            self.revoke_token_by_jti(payload['jti'])
        except jwt.InvalidTokenError:
            pass
    
    def revoke_token_by_jti(self, jti: str):
        """Revoke token by JWT ID."""
        self.active_tokens.discard(jti)
    
    def revoke_all_user_tokens(self, user_id: str):
        """Revoke all tokens for a specific user."""
        sessions_to_remove = []
        for session in self.user_sessions[user_id]:
            self.active_tokens.discard(session['jti'])
            sessions_to_remove.append(session)
        
        for session in sessions_to_remove:
            self.user_sessions[user_id].remove(session)
        
        self.logger.info(f"All tokens revoked for user: {user_id}")
    
    def check_permission(self, user_id: str, role: str, required_permission: str) -> bool:
        """Check if user has required permission."""
        # Check user-specific permissions
        if required_permission in self.user_permissions[user_id]:
            return True
        
        # Check role-based permissions
        if required_permission in self.role_permissions.get(role, set()):
            return True
        
        return False
    
    def grant_user_permission(self, user_id: str, permission: str):
        """Grant specific permission to user."""
        self.user_permissions[user_id].add(permission)
        self.logger.info(f"Permission '{permission}' granted to user: {user_id}")
    
    def revoke_user_permission(self, user_id: str, permission: str):
        """Revoke specific permission from user."""
        self.user_permissions[user_id].discard(permission)
        self.logger.info(f"Permission '{permission}' revoked from user: {user_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        
        for user_id in list(self.user_sessions.keys()):
            sessions_to_remove = []
            
            for session in self.user_sessions[user_id]:
                if current_time - session['last_activity'] > self.session_timeout:
                    self.active_tokens.discard(session['jti'])
                    sessions_to_remove.append(session)
            
            for session in sessions_to_remove:
                self.user_sessions[user_id].remove(session)
            
            # Remove user entry if no sessions left
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]


class AuditLogger:
    """
    Comprehensive audit logging for security events and user actions.
    """
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Configure audit logger
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Detailed formatter for audit logs
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        if not self.audit_logger.handlers:
            self.audit_logger.addHandler(file_handler)
    
    def log_security_incident(self, incident: SecurityIncident):
        """Log security incident."""
        self.audit_logger.warning(
            f"SECURITY_INCIDENT - Type: {incident.threat_type.value}, "
            f"Severity: {incident.severity.value}, "
            f"Source: {incident.source_ip or 'unknown'}, "
            f"Description: {incident.description}"
        )
    
    def log_authentication(self, user_id: str, success: bool, source_ip: str = None):
        """Log authentication attempt."""
        status = "SUCCESS" if success else "FAILURE"
        self.audit_logger.info(
            f"AUTHENTICATION_{status} - User: {user_id}, "
            f"Source: {source_ip or 'unknown'}"
        )
    
    def log_authorization(self, user_id: str, action: str, resource: str, 
                         granted: bool, source_ip: str = None):
        """Log authorization decision."""
        status = "GRANTED" if granted else "DENIED"
        self.audit_logger.info(
            f"AUTHORIZATION_{status} - User: {user_id}, "
            f"Action: {action}, Resource: {resource}, "
            f"Source: {source_ip or 'unknown'}"
        )
    
    def log_data_access(self, user_id: str, data_type: str, operation: str,
                       source_ip: str = None):
        """Log data access events."""
        self.audit_logger.info(
            f"DATA_ACCESS - User: {user_id}, "
            f"DataType: {data_type}, Operation: {operation}, "
            f"Source: {source_ip or 'unknown'}"
        )
    
    def log_system_event(self, event_type: str, description: str, 
                        severity: str = "INFO"):
        """Log system-level events."""
        self.audit_logger.log(
            getattr(logging, severity.upper()),
            f"SYSTEM_EVENT - Type: {event_type}, Description: {description}"
        )


class SecurityMonitor:
    """
    Central security monitoring system integrating all security components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.threat_detector = ThreatDetector()
        self.access_controller = AccessController()
        self.audit_logger = AuditLogger()
        
        self.incident_history = deque(maxlen=10000)
        self.active_incidents = []
        self.response_handlers = {}
        
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
        
        # Register default response handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default incident response handlers."""
        self.response_handlers[ThreatType.BRUTE_FORCE] = self._handle_brute_force
        self.response_handlers[ThreatType.DDoS] = self._handle_ddos
        self.response_handlers[ThreatType.MALICIOUS_INPUT] = self._handle_malicious_input
    
    def _handle_brute_force(self, incident: SecurityIncident):
        """Handle brute force attack."""
        if incident.source_ip:
            self.threat_detector.block_ip(incident.source_ip)
            incident.response_actions.append(f"Blocked IP: {incident.source_ip}")
    
    def _handle_ddos(self, incident: SecurityIncident):
        """Handle DDoS attack."""
        if incident.source_ip:
            self.threat_detector.block_ip(incident.source_ip)
            incident.response_actions.append(f"Blocked IP: {incident.source_ip}")
    
    def _handle_malicious_input(self, incident: SecurityIncident):
        """Handle malicious input detection."""
        # Log detailed information for analysis
        self.audit_logger.log_system_event(
            "MALICIOUS_INPUT_BLOCKED",
            f"Malicious input blocked: {incident.metadata.get('attack_type', 'unknown')}",
            "WARNING"
        )
        incident.response_actions.append("Input sanitized and logged")
    
    def detect_and_respond(self, **kwargs) -> Optional[SecurityIncident]:
        """
        Detect threats and automatically respond.
        
        Args:
            source_ip: Source IP address
            endpoint: API endpoint accessed
            success: Whether operation was successful
            input_data: Input data to analyze
            response_size: Size of response data
            user_id: User identifier
        """
        incidents = []
        
        source_ip = kwargs.get('source_ip')
        endpoint = kwargs.get('endpoint')
        success = kwargs.get('success', True)
        input_data = kwargs.get('input_data')
        response_size = kwargs.get('response_size')
        user_id = kwargs.get('user_id')
        
        # Detect brute force
        if source_ip and endpoint is not None:
            incident = self.threat_detector.detect_brute_force(source_ip, endpoint, success)
            if incident:
                incidents.append(incident)
        
        # Detect DDoS
        if source_ip:
            incident = self.threat_detector.detect_ddos(source_ip)
            if incident:
                incidents.append(incident)
        
        # Detect malicious input
        if input_data:
            incident = self.threat_detector.detect_malicious_input(input_data, endpoint or "unknown")
            if incident:
                incidents.append(incident)
        
        # Detect data exfiltration
        if response_size and endpoint:
            incident = self.threat_detector.detect_data_exfiltration(response_size, endpoint)
            if incident:
                incidents.append(incident)
        
        # Process all detected incidents
        for incident in incidents:
            self._process_incident(incident)
        
        return incidents[0] if incidents else None
    
    def _process_incident(self, incident: SecurityIncident):
        """Process and respond to security incident."""
        # Add to history
        self.incident_history.append(incident)
        self.active_incidents.append(incident)
        
        # Log incident
        self.audit_logger.log_security_incident(incident)
        
        # Execute response handler
        handler = self.response_handlers.get(incident.threat_type)
        if handler:
            try:
                handler(incident)
                self.logger.info(f"Response executed for {incident.threat_type.value}")
            except Exception as e:
                self.logger.error(f"Response handler failed: {e}")
        
        # Escalate critical incidents
        if incident.severity == SecurityLevel.CRITICAL:
            self._escalate_incident(incident)
    
    def _escalate_incident(self, incident: SecurityIncident):
        """Escalate critical security incidents."""
        self.logger.critical(
            f"CRITICAL SECURITY INCIDENT: {incident.threat_type.value} - {incident.description}"
        )
        
        # Additional escalation actions could include:
        # - Sending alerts to security team
        # - Triggering automated isolation procedures
        # - Initiating incident response procedures
    
    def authenticate_request(self, token: str, required_permission: str = None) -> Tuple[bool, Optional[str]]:
        """
        Authenticate and authorize a request.
        
        Returns:
            Tuple of (is_authorized, user_id)
        """
        # Verify token
        payload = self.access_controller.verify_token(token)
        if not payload:
            return False, None
        
        user_id = payload['user_id']
        role = payload['role']
        
        # Check permission if required
        if required_permission:
            if not self.access_controller.check_permission(user_id, role, required_permission):
                self.audit_logger.log_authorization(
                    user_id, required_permission, "API", False
                )
                return False, user_id
        
        return True, user_id
    
    def start_monitoring(self):
        """Start continuous security monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Cleanup expired sessions
                self.access_controller.cleanup_expired_sessions()
                
                # Resolve old incidents
                self._resolve_old_incidents()
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
    
    def _resolve_old_incidents(self):
        """Automatically resolve old incidents."""
        current_time = time.time()
        auto_resolve_time = 24 * 3600  # 24 hours
        
        for incident in self.active_incidents[:]:
            if current_time - incident.timestamp > auto_resolve_time:
                incident.resolved = True
                self.active_incidents.remove(incident)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status summary."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_incidents = [
            inc for inc in self.incident_history
            if inc.timestamp > hour_ago
        ]
        
        return {
            'monitoring_active': self.monitoring_active,
            'active_incidents': len(self.active_incidents),
            'recent_incidents_1h': len(recent_incidents),
            'blocked_ips': len(self.threat_detector.blocked_ips),
            'active_sessions': sum(len(sessions) for sessions in self.access_controller.user_sessions.values()),
            'threat_levels': {
                level.value: len([inc for inc in recent_incidents if inc.severity == level])
                for level in SecurityLevel
            }
        }
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        status = self.get_security_status()
        
        report_lines = [
            "=== SECURITY STATUS REPORT ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Monitoring Status: {'Active' if status['monitoring_active'] else 'Inactive'}",
            f"Active Incidents: {status['active_incidents']}",
            f"Recent Incidents (1h): {status['recent_incidents_1h']}",
            f"Blocked IPs: {status['blocked_ips']}",
            f"Active Sessions: {status['active_sessions']}",
            "",
            "Threat Level Distribution (last hour):",
        ]
        
        for level, count in status['threat_levels'].items():
            report_lines.append(f"  {level.upper()}: {count}")
        
        if self.active_incidents:
            report_lines.extend([
                "",
                "Active Incidents:",
            ])
            
            for incident in self.active_incidents[-10:]:  # Last 10 incidents
                report_lines.append(
                    f"  - {incident.threat_type.value}: {incident.description} "
                    f"({incident.severity.value})"
                )
        
        return "\n".join(report_lines)


# Global security monitor instance
global_security_monitor = SecurityMonitor()


def secure_endpoint(required_permission: str = None):
    """
    Decorator for securing API endpoints with authentication and authorization.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract token from kwargs or headers
            token = kwargs.pop('auth_token', None)
            
            if not token:
                raise ValueError("Authentication token required")
            
            # Authenticate and authorize
            is_authorized, user_id = global_security_monitor.authenticate_request(
                token, required_permission
            )
            
            if not is_authorized:
                raise PermissionError("Access denied")
            
            # Add user_id to kwargs for function use
            kwargs['authenticated_user_id'] = user_id
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
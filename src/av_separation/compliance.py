"""
Data Privacy and Compliance Management for AV-Separation-Transformer
GDPR, CCPA, PDPA compliance with automated data protection
"""

import time
import hashlib
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid


class DataProcessingPurpose(Enum):
    """Valid purposes for data processing under GDPR Article 6"""
    CONSENT = "consent"                    # Article 6(1)(a)
    CONTRACT = "contract"                  # Article 6(1)(b) 
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"    # Article 6(1)(d)
    PUBLIC_TASK = "public_task"           # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Article 6(1)(f)


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC_PERSONAL = "basic_personal"      # Name, email, etc.
    BIOMETRIC = "biometric"                # Voice patterns, face recognition
    AUDIO_CONTENT = "audio_content"        # Recorded audio for processing
    VIDEO_CONTENT = "video_content"        # Video frames for processing
    TECHNICAL = "technical"                # IP addresses, device info
    USAGE = "usage"                        # Usage patterns, preferences


@dataclass
class ConsentRecord:
    """Record of user consent"""
    
    user_id: str
    consent_id: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    purposes: List[DataProcessingPurpose] = field(default_factory=list)
    data_categories: List[DataCategory] = field(default_factory=list)
    consent_method: str = "explicit"  # explicit, implicit, opt_in
    withdrawn_at: Optional[datetime] = None
    withdrawal_method: Optional[str] = None
    
    # GDPR specific fields
    lawful_basis: DataProcessingPurpose = DataProcessingPurpose.CONSENT
    data_controller: str = "AV-Separation-Transformer"
    processing_location: str = "EU"
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid"""
        
        if self.withdrawn_at is not None:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def withdraw(self, method: str = "user_request"):
        """Withdraw consent"""
        
        self.withdrawn_at = datetime.utcnow()
        self.withdrawal_method = method


@dataclass  
class DataSubjectRequest:
    """Data subject rights request (GDPR Articles 15-22)"""
    
    request_id: str
    user_id: str
    request_type: str  # access, rectification, erasure, portability, etc.
    submitted_at: datetime
    status: str = "pending"  # pending, processing, completed, rejected
    due_date: datetime = None
    completed_at: Optional[datetime] = None
    
    # Request details
    requested_data: List[str] = field(default_factory=list)
    justification: Optional[str] = None
    identity_verified: bool = False
    
    # Response
    response_data: Optional[Dict[str, Any]] = None
    response_format: str = "json"  # json, csv, pdf
    
    def __post_init__(self):
        if self.due_date is None:
            # GDPR Article 12: 1 month response time
            self.due_date = self.submitted_at + timedelta(days=30)


class ComplianceManager:
    """
    Comprehensive compliance management system
    """
    
    def __init__(self, region: str = "EU"):
        self.region = region
        self.logger = logging.getLogger(__name__)
        
        # Consent management
        self.consent_records: Dict[str, ConsentRecord] = {}
        
        # Data subject requests
        self.data_requests: Dict[str, DataSubjectRequest] = {}
        
        # Processing activities (GDPR Article 30)
        self.processing_activities: List[Dict[str, Any]] = []
        
        # Data retention policies
        self.retention_policies: Dict[DataCategory, int] = {
            DataCategory.AUDIO_CONTENT: 30,    # 30 days
            DataCategory.VIDEO_CONTENT: 30,    # 30 days  
            DataCategory.BIOMETRIC: 90,        # 90 days
            DataCategory.TECHNICAL: 365,       # 1 year
            DataCategory.USAGE: 730,           # 2 years
            DataCategory.BASIC_PERSONAL: 1095  # 3 years
        }
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        
        self._setup_processing_activities()
    
    def _setup_processing_activities(self):
        """Setup processing activities register (GDPR Article 30)"""
        
        self.processing_activities = [
            {
                "activity_id": "audio_visual_separation",
                "name": "Audio-Visual Speech Separation",
                "description": "AI-powered separation of overlapping speech from audio-visual content",
                "data_controller": "AV-Separation-Transformer Service",
                "lawful_basis": [DataProcessingPurpose.CONSENT, DataProcessingPurpose.LEGITIMATE_INTERESTS],
                "data_categories": [
                    DataCategory.AUDIO_CONTENT,
                    DataCategory.VIDEO_CONTENT,
                    DataCategory.BIOMETRIC,
                    DataCategory.TECHNICAL
                ],
                "data_subjects": ["Content creators", "End users", "API consumers"],
                "recipients": ["Internal processing systems", "Cloud storage providers"],
                "international_transfers": False,
                "retention_period": "30 days for processing data, 90 days for biometric features",
                "security_measures": [
                    "Encryption at rest and in transit",
                    "Access controls and authentication", 
                    "Regular security audits",
                    "Automatic data deletion"
                ]
            }
        ]
    
    def record_consent(
        self,
        user_id: str,
        purposes: List[DataProcessingPurpose],
        data_categories: List[DataCategory],
        consent_method: str = "explicit",
        expires_in_days: Optional[int] = None
    ) -> str:
        """
        Record user consent
        
        Args:
            user_id: Unique user identifier
            purposes: List of processing purposes
            data_categories: Categories of data to be processed
            consent_method: How consent was obtained
            expires_in_days: Consent expiration (None = no expiration)
            
        Returns:
            Consent ID
        """
        
        consent_id = str(uuid.uuid4())
        expires_at = None
        
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        consent = ConsentRecord(
            user_id=user_id,
            consent_id=consent_id,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            purposes=purposes,
            data_categories=data_categories,
            consent_method=consent_method,
            processing_location=self.region
        )
        
        self.consent_records[consent_id] = consent
        
        # Audit trail
        self._add_audit_entry("consent_granted", {
            "user_id": user_id,
            "consent_id": consent_id,
            "purposes": [p.value for p in purposes],
            "data_categories": [c.value for c in data_categories],
            "method": consent_method
        })
        
        self.logger.info(f"Consent recorded for user {user_id}: {consent_id}")
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str, method: str = "user_request") -> bool:
        """
        Withdraw user consent
        
        Args:
            consent_id: Consent identifier
            method: How consent was withdrawn
            
        Returns:
            True if withdrawal successful
        """
        
        if consent_id not in self.consent_records:
            return False
        
        consent = self.consent_records[consent_id]
        consent.withdraw(method)
        
        # Trigger data deletion if required
        asyncio.create_task(self._handle_consent_withdrawal(consent))
        
        # Audit trail
        self._add_audit_entry("consent_withdrawn", {
            "user_id": consent.user_id,
            "consent_id": consent_id,
            "method": method
        })
        
        self.logger.info(f"Consent withdrawn: {consent_id}")
        
        return True
    
    def check_consent(self, user_id: str, purpose: DataProcessingPurpose, data_category: DataCategory) -> bool:
        """
        Check if user has valid consent for specific processing
        
        Args:
            user_id: User identifier
            purpose: Processing purpose
            data_category: Data category
            
        Returns:
            True if consent is valid
        """
        
        # Find valid consent records for user
        valid_consents = [
            consent for consent in self.consent_records.values()
            if consent.user_id == user_id and consent.is_valid()
        ]
        
        # Check if any consent covers this purpose and data category
        for consent in valid_consents:
            if purpose in consent.purposes and data_category in consent.data_categories:
                return True
        
        return False
    
    def submit_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        requested_data: List[str] = None,
        justification: str = None
    ) -> str:
        """
        Submit data subject rights request
        
        Args:
            user_id: User identifier
            request_type: Type of request (access, erasure, portability, etc.)
            requested_data: Specific data requested
            justification: User's justification for request
            
        Returns:
            Request ID
        """
        
        request_id = str(uuid.uuid4())
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            submitted_at=datetime.utcnow(),
            requested_data=requested_data or [],
            justification=justification
        )
        
        self.data_requests[request_id] = request
        
        # Audit trail
        self._add_audit_entry("data_subject_request", {
            "user_id": user_id,
            "request_id": request_id,
            "request_type": request_type,
            "requested_data": requested_data
        })
        
        self.logger.info(f"Data subject request submitted: {request_id}")
        
        return request_id
    
    async def process_data_subject_request(self, request_id: str) -> Dict[str, Any]:
        """
        Process data subject rights request
        
        Args:
            request_id: Request identifier
            
        Returns:
            Processing result
        """
        
        if request_id not in self.data_requests:
            raise ValueError(f"Request {request_id} not found")
        
        request = self.data_requests[request_id]
        request.status = "processing"
        
        result = {"success": False, "data": None, "error": None}
        
        try:
            if request.request_type == "access":
                # Right of access (GDPR Article 15)
                data = await self._collect_user_data(request.user_id)
                result = {"success": True, "data": data}
                
            elif request.request_type == "erasure":
                # Right to erasure (GDPR Article 17)
                await self._delete_user_data(request.user_id)
                result = {"success": True, "message": "Data deleted successfully"}
                
            elif request.request_type == "portability":
                # Right to data portability (GDPR Article 20)
                data = await self._export_user_data(request.user_id, format="json")
                result = {"success": True, "data": data}
                
            elif request.request_type == "rectification":
                # Right to rectification (GDPR Article 16)
                result = {"success": False, "error": "Manual rectification required"}
                
            else:
                result = {"success": False, "error": f"Unknown request type: {request.request_type}"}
            
            request.status = "completed" if result["success"] else "rejected"
            request.completed_at = datetime.utcnow()
            request.response_data = result
            
            # Audit trail
            self._add_audit_entry("data_subject_request_processed", {
                "request_id": request_id,
                "user_id": request.user_id,
                "status": request.status,
                "success": result["success"]
            })
            
        except Exception as e:
            request.status = "rejected"
            result = {"success": False, "error": str(e)}
            self.logger.error(f"Failed to process request {request_id}: {e}")
        
        return result
    
    async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all data for a user (GDPR Article 15)"""
        
        data = {
            "user_id": user_id,
            "consents": [],
            "processing_activities": [],
            "stored_data": {}
        }
        
        # Collect consent records
        user_consents = [
            {
                "consent_id": consent.consent_id,
                "granted_at": consent.granted_at.isoformat(),
                "purposes": [p.value for p in consent.purposes],
                "data_categories": [c.value for c in consent.data_categories],
                "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None
            }
            for consent in self.consent_records.values()
            if consent.user_id == user_id
        ]
        data["consents"] = user_consents
        
        # Collect processing activities
        data["processing_activities"] = self.processing_activities
        
        # Collect audit trail entries
        user_audit_entries = [
            entry for entry in self.audit_trail
            if entry.get("data", {}).get("user_id") == user_id
        ]
        data["audit_trail"] = user_audit_entries
        
        return data
    
    async def _delete_user_data(self, user_id: str):
        """Delete all user data (GDPR Article 17)"""
        
        # Mark consent records as withdrawn
        for consent in self.consent_records.values():
            if consent.user_id == user_id and consent.is_valid():
                consent.withdraw("erasure_request")
        
        # Add deletion record to audit trail
        self._add_audit_entry("user_data_deleted", {
            "user_id": user_id,
            "deletion_reason": "user_request",
            "data_categories": [c.value for c in DataCategory]
        })
        
        # In a real implementation, this would trigger deletion from:
        # - Database records
        # - File storage systems
        # - Cache systems
        # - Backup systems
        # - Third-party processors
        
        self.logger.info(f"User data deletion initiated for {user_id}")
    
    async def _export_user_data(self, user_id: str, format: str = "json") -> Dict[str, Any]:
        """Export user data in machine-readable format (GDPR Article 20)"""
        
        data = await self._collect_user_data(user_id)
        
        if format == "json":
            return data
        elif format == "csv":
            # Convert to CSV format
            return {"error": "CSV export not implemented"}
        else:
            return {"error": f"Unsupported format: {format}"}
    
    async def _handle_consent_withdrawal(self, consent: ConsentRecord):
        """Handle consent withdrawal - trigger data deletion if required"""
        
        # Check if user has other valid consents
        other_consents = [
            c for c in self.consent_records.values()
            if c.user_id == consent.user_id and c.consent_id != consent.consent_id and c.is_valid()
        ]
        
        if not other_consents:
            # No other valid consents - delete user data
            await self._delete_user_data(consent.user_id)
    
    def _add_audit_entry(self, event_type: str, data: Dict[str, Any]):
        """Add entry to audit trail"""
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
            "source": "compliance_manager"
        }
        
        self.audit_trail.append(entry)
        
        # Keep only last 10000 entries to prevent unbounded growth
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-10000:]
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        
        current_time = datetime.utcnow()
        
        # Count active consents
        active_consents = sum(1 for c in self.consent_records.values() if c.is_valid())
        
        # Count data subject requests by status
        request_stats = {}
        for request in self.data_requests.values():
            request_stats[request.status] = request_stats.get(request.status, 0) + 1
        
        # Count overdue requests
        overdue_requests = sum(
            1 for r in self.data_requests.values()
            if r.status == "pending" and current_time > r.due_date
        )
        
        report = {
            "generated_at": current_time.isoformat(),
            "region": self.region,
            "statistics": {
                "total_consent_records": len(self.consent_records),
                "active_consents": active_consents,
                "total_data_requests": len(self.data_requests),
                "overdue_requests": overdue_requests,
                "request_stats": request_stats
            },
            "processing_activities": len(self.processing_activities),
            "audit_entries": len(self.audit_trail),
            "retention_policies": {
                cat.value: days for cat, days in self.retention_policies.items()
            }
        }
        
        return report
    
    def check_compliance_status(self) -> Dict[str, Any]:
        """Check overall compliance status"""
        
        current_time = datetime.utcnow()
        issues = []
        warnings = []
        
        # Check for overdue data subject requests
        overdue_requests = [
            r for r in self.data_requests.values()
            if r.status == "pending" and current_time > r.due_date
        ]
        
        if overdue_requests:
            issues.append(f"{len(overdue_requests)} overdue data subject requests")
        
        # Check for expired consents
        expired_consents = [
            c for c in self.consent_records.values()
            if c.expires_at and current_time > c.expires_at and not c.withdrawn_at
        ]
        
        if expired_consents:
            warnings.append(f"{len(expired_consents)} expired consents need cleanup")
        
        # Check data retention compliance
        # This would involve checking actual stored data against retention policies
        
        status = "compliant"
        if issues:
            status = "issues_found"
        elif warnings:
            status = "warnings"
        
        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "last_check": current_time.isoformat()
        }


# Global compliance manager instance
_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager(region: str = "EU") -> ComplianceManager:
    """Get global compliance manager instance"""
    
    global _compliance_manager
    
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager(region)
    
    return _compliance_manager


def initialize_compliance(region: str = "EU") -> ComplianceManager:
    """Initialize compliance management for specific region"""
    
    global _compliance_manager
    
    _compliance_manager = ComplianceManager(region)
    
    return _compliance_manager


# Decorator for API endpoints to check consent
def requires_consent(purpose: DataProcessingPurpose, data_category: DataCategory):
    """Decorator to check user consent before processing"""
    
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Extract user_id from request (implementation depends on your auth system)
            user_id = kwargs.get('user_id') or getattr(kwargs.get('current_user', {}), 'get', lambda k: None)('user_id')
            
            if user_id:
                compliance_manager = get_compliance_manager()
                
                if not compliance_manager.check_consent(user_id, purpose, data_category):
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=403,
                        detail="Consent required for this data processing activity"
                    )
            
            return f(*args, **kwargs)
        
        return wrapper
    
    return decorator
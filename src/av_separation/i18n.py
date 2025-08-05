"""
Internationalization (i18n) Support for AV-Separation-Transformer
Multi-language support with compliance-aware messaging
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SupportedLanguage(Enum):
    """Supported languages for i18n"""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


@dataclass
class RegionConfig:
    """Configuration for regional compliance"""
    
    region_code: str
    gdpr_required: bool = False
    ccpa_required: bool = False
    pdpa_required: bool = False
    data_residency_required: bool = False
    audit_logging_required: bool = True
    encryption_required: bool = True
    max_data_retention_days: int = 365
    
    # Regional-specific requirements
    additional_compliance: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_compliance is None:
            self.additional_compliance = {}


# Regional compliance configurations
REGION_CONFIGS = {
    "EU": RegionConfig(
        region_code="EU",
        gdpr_required=True,
        data_residency_required=True,
        max_data_retention_days=730,  # GDPR allows longer retention with consent
        additional_compliance={
            "right_to_be_forgotten": True,
            "consent_management": True,
            "data_portability": True,
            "privacy_by_design": True
        }
    ),
    "US-CA": RegionConfig(
        region_code="US-CA", 
        ccpa_required=True,
        max_data_retention_days=365,
        additional_compliance={
            "do_not_sell": True,
            "consumer_rights": True,
            "opt_out_mechanisms": True
        }
    ),
    "SG": RegionConfig(
        region_code="SG",
        pdpa_required=True,
        data_residency_required=True,
        max_data_retention_days=365,
        additional_compliance={
            "data_breach_notification": True,
            "consent_withdrawal": True
        }
    ),
    "US": RegionConfig(
        region_code="US",
        max_data_retention_days=1095,  # 3 years
        additional_compliance={
            "section_508_compliance": True,  # Accessibility
            "ferpa_compliance": False  # Educational records
        }
    ),
    "UK": RegionConfig(
        region_code="UK",
        gdpr_required=True,  # UK GDPR
        data_residency_required=True,
        max_data_retention_days=730,
        additional_compliance={
            "uk_gdpr": True,
            "ico_compliance": True
        }
    ),
    "CA": RegionConfig(
        region_code="CA",
        max_data_retention_days=730,
        additional_compliance={
            "pipeda_compliance": True,
            "french_language_support": True
        }
    ),
    "JP": RegionConfig(
        region_code="JP",
        data_residency_required=True,
        max_data_retention_days=365,
        additional_compliance={
            "appi_compliance": True,  # Act on Protection of Personal Information
            "ppc_compliance": True    # Personal Information Protection Commission
        }
    ),
    "AU": RegionConfig(
        region_code="AU",
        max_data_retention_days=730,
        additional_compliance={
            "privacy_act_compliance": True,
            "notifiable_data_breach": True
        }
    )
}


class LocalizationManager:
    """
    Manages localized strings and regional compliance messages
    """
    
    def __init__(self, default_language: str = "en", default_region: str = "US"):
        self.default_language = default_language
        self.default_region = default_region
        self.current_language = default_language
        self.current_region = default_region
        
        self.translations: Dict[str, Dict[str, str]] = {}
        self.region_config = REGION_CONFIGS.get(default_region, REGION_CONFIGS["US"])
        
        self.logger = logging.getLogger(__name__)
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files"""
        
        # Default English translations
        self.translations["en"] = {
            # API Messages
            "api.separation.success": "Audio-visual separation completed successfully",
            "api.separation.failed": "Separation processing failed",
            "api.model.loading": "Loading audio-visual separation model",
            "api.model.loaded": "Model loaded successfully",
            "api.validation.invalid_file": "Invalid file format",
            "api.validation.file_too_large": "File size exceeds maximum limit",
            "api.validation.malicious_filename": "Potentially malicious filename detected",
            "api.auth.invalid_credentials": "Invalid authentication credentials",
            "api.auth.access_denied": "Access denied - insufficient permissions",
            "api.rate_limit.exceeded": "Rate limit exceeded - please try again later",
            
            # Processing Messages
            "processing.started": "Processing started",
            "processing.audio_loading": "Loading audio data",
            "processing.video_loading": "Loading video data", 
            "processing.separating": "Separating audio sources",
            "processing.completed": "Processing completed",
            "processing.error": "Processing error occurred",
            
            # Error Messages
            "error.internal": "Internal server error",
            "error.not_found": "Resource not found",
            "error.timeout": "Request timeout",
            "error.memory": "Insufficient memory",
            "error.gpu": "GPU processing error",
            
            # Privacy & Compliance
            "privacy.data_processing": "Your data is being processed in accordance with our privacy policy",
            "privacy.data_retention": "Data will be retained for {days} days as per regional requirements",
            "privacy.gdpr_notice": "Under GDPR, you have the right to access, rectify, and delete your personal data",
            "privacy.ccpa_notice": "Under CCPA, you have the right to know what personal information is collected and opt-out of sale",
            "privacy.consent_required": "Processing requires your explicit consent",
            "privacy.data_transfer": "Data may be transferred to secure processing facilities",
            
            # Performance Messages
            "performance.optimizing": "Optimizing model for better performance",
            "performance.caching": "Caching results for faster future processing",
            "performance.scaling": "Auto-scaling resources based on demand",
            
            # Accessibility
            "accessibility.audio_description": "Audio separation in progress",
            "accessibility.processing_status": "Processing status: {status}",
            "accessibility.results_ready": "Separation results are ready for download"
        }
        
        # Spanish translations
        self.translations["es"] = {
            "api.separation.success": "Separación audio-visual completada exitosamente",
            "api.separation.failed": "Falló el procesamiento de separación",
            "api.model.loading": "Cargando modelo de separación audio-visual",
            "api.model.loaded": "Modelo cargado exitosamente",
            "api.validation.invalid_file": "Formato de archivo inválido",
            "api.validation.file_too_large": "El tamaño del archivo excede el límite máximo",
            "api.validation.malicious_filename": "Nombre de archivo potencialmente malicioso detectado",
            "api.auth.invalid_credentials": "Credenciales de autenticación inválidas",
            "api.auth.access_denied": "Acceso denegado - permisos insuficientes",
            "api.rate_limit.exceeded": "Límite de velocidad excedido - intente nuevamente más tarde",
            
            "processing.started": "Procesamiento iniciado",
            "processing.audio_loading": "Cargando datos de audio",
            "processing.video_loading": "Cargando datos de video",
            "processing.separating": "Separando fuentes de audio", 
            "processing.completed": "Procesamiento completado",
            "processing.error": "Ocurrió un error de procesamiento",
            
            "error.internal": "Error interno del servidor",
            "error.not_found": "Recurso no encontrado",
            "error.timeout": "Tiempo de espera agotado",
            "error.memory": "Memoria insuficiente",
            "error.gpu": "Error de procesamiento GPU",
            
            "privacy.data_processing": "Sus datos están siendo procesados de acuerdo con nuestra política de privacidad",
            "privacy.data_retention": "Los datos se conservarán por {days} días según los requisitos regionales",
            "privacy.gdpr_notice": "Bajo GDPR, tiene derecho a acceder, rectificar y eliminar sus datos personales",
            "privacy.consent_required": "El procesamiento requiere su consentimiento explícito",
            "privacy.data_transfer": "Los datos pueden transferirse a instalaciones de procesamiento seguras"
        }
        
        # French translations
        self.translations["fr"] = {
            "api.separation.success": "Séparation audio-visuelle terminée avec succès",
            "api.separation.failed": "Échec du traitement de séparation",
            "api.model.loading": "Chargement du modèle de séparation audio-visuelle",
            "api.model.loaded": "Modèle chargé avec succès",
            "api.validation.invalid_file": "Format de fichier invalide",
            "api.validation.file_too_large": "La taille du fichier dépasse la limite maximale",
            "api.validation.malicious_filename": "Nom de fichier potentiellement malveillant détecté",
            "api.auth.invalid_credentials": "Identifiants d'authentification invalides",
            "api.auth.access_denied": "Accès refusé - permissions insuffisantes",
            "api.rate_limit.exceeded": "Limite de débit dépassée - veuillez réessayer plus tard",
            
            "processing.started": "Traitement commencé",
            "processing.audio_loading": "Chargement des données audio",
            "processing.video_loading": "Chargement des données vidéo",
            "processing.separating": "Séparation des sources audio",
            "processing.completed": "Traitement terminé",
            "processing.error": "Erreur de traitement survenue",
            
            "privacy.data_processing": "Vos données sont traitées conformément à notre politique de confidentialité",
            "privacy.data_retention": "Les données seront conservées pendant {days} jours selon les exigences régionales",
            "privacy.gdpr_notice": "Sous le RGPD, vous avez le droit d'accéder, rectifier et supprimer vos données personnelles",
            "privacy.consent_required": "Le traitement nécessite votre consentement explicite",
            "privacy.data_transfer": "Les données peuvent être transférées vers des installations de traitement sécurisées"
        }
        
        # German translations
        self.translations["de"] = {
            "api.separation.success": "Audio-visuelle Trennung erfolgreich abgeschlossen",
            "api.separation.failed": "Trennungsverarbeitung fehlgeschlagen",
            "api.model.loading": "Lade Audio-visuelles Trennungsmodell",
            "api.model.loaded": "Modell erfolgreich geladen",
            "api.validation.invalid_file": "Ungültiges Dateiformat",
            "api.validation.file_too_large": "Dateigröße überschreitet maximale Grenze",
            "api.validation.malicious_filename": "Potenziell bösartiger Dateiname erkannt",
            "api.auth.invalid_credentials": "Ungültige Authentifizierungsdaten",
            "api.auth.access_denied": "Zugriff verweigert - unzureichende Berechtigungen",
            "api.rate_limit.exceeded": "Rate-Limit überschritten - bitte später erneut versuchen",
            
            "processing.started": "Verarbeitung gestartet",
            "processing.audio_loading": "Lade Audiodaten",
            "processing.video_loading": "Lade Videodaten",
            "processing.separating": "Trenne Audioquellen",
            "processing.completed": "Verarbeitung abgeschlossen",
            "processing.error": "Verarbeitungsfehler aufgetreten",
            
            "privacy.data_processing": "Ihre Daten werden gemäß unserer Datenschutzrichtlinie verarbeitet",
            "privacy.data_retention": "Daten werden für {days} Tage gemäß regionalen Anforderungen gespeichert",
            "privacy.gdpr_notice": "Unter der DSGVO haben Sie das Recht, auf Ihre personenbezogenen Daten zuzugreifen, sie zu berichtigen und zu löschen",
            "privacy.consent_required": "Die Verarbeitung erfordert Ihre ausdrückliche Zustimmung",
            "privacy.data_transfer": "Daten können an sichere Verarbeitungseinrichtungen übertragen werden"
        }
        
        # Japanese translations
        self.translations["ja"] = {
            "api.separation.success": "音声映像分離が正常に完了しました",
            "api.separation.failed": "分離処理に失敗しました",
            "api.model.loading": "音声映像分離モデルを読み込み中",
            "api.model.loaded": "モデルが正常に読み込まれました",
            "api.validation.invalid_file": "無効なファイル形式",
            "api.validation.file_too_large": "ファイルサイズが最大制限を超えています",
            "api.validation.malicious_filename": "潜在的に悪意のあるファイル名が検出されました",
            "api.auth.invalid_credentials": "認証資格情報が無効です",
            "api.auth.access_denied": "アクセス拒否 - 権限が不十分です",
            "api.rate_limit.exceeded": "レート制限を超えました - 後でもう一度お試しください",
            
            "processing.started": "処理開始",
            "processing.audio_loading": "音声データを読み込み中",
            "processing.video_loading": "映像データを読み込み中",
            "processing.separating": "音声ソースを分離中",
            "processing.completed": "処理完了",
            "processing.error": "処理エラーが発生しました",
            
            "privacy.data_processing": "お客様のデータはプライバシーポリシーに従って処理されます",
            "privacy.data_retention": "地域要件に従ってデータは{days}日間保持されます",
            "privacy.consent_required": "処理にはお客様の明示的な同意が必要です",
            "privacy.data_transfer": "データは安全な処理施設に転送される場合があります"
        }
        
        # Chinese Simplified translations
        self.translations["zh-CN"] = {
            "api.separation.success": "音视频分离成功完成", 
            "api.separation.failed": "分离处理失败",
            "api.model.loading": "正在加载音视频分离模型",
            "api.model.loaded": "模型加载成功",
            "api.validation.invalid_file": "无效的文件格式",
            "api.validation.file_too_large": "文件大小超出最大限制",
            "api.validation.malicious_filename": "检测到潜在恶意文件名",
            "api.auth.invalid_credentials": "身份验证凭据无效",
            "api.auth.access_denied": "访问被拒绝 - 权限不足",
            "api.rate_limit.exceeded": "超出速率限制 - 请稍后重试",
            
            "processing.started": "处理开始",
            "processing.audio_loading": "正在加载音频数据",
            "processing.video_loading": "正在加载视频数据",
            "processing.separating": "正在分离音频源",
            "processing.completed": "处理完成", 
            "processing.error": "发生处理错误",
            
            "privacy.data_processing": "您的数据正在根据我们的隐私政策进行处理",
            "privacy.data_retention": "根据地区要求，数据将保留{days}天",
            "privacy.consent_required": "处理需要您的明确同意",
            "privacy.data_transfer": "数据可能会传输到安全的处理设施"
        }
        
        self.logger.info(f"Loaded translations for {len(self.translations)} languages")
    
    def set_language(self, language_code: str):
        """Set current language"""
        
        if language_code in self.translations:
            self.current_language = language_code
            self.logger.info(f"Language set to {language_code}")
        else:
            self.logger.warning(f"Language {language_code} not supported, using {self.default_language}")
            self.current_language = self.default_language
    
    def set_region(self, region_code: str):
        """Set current region and update compliance requirements"""
        
        if region_code in REGION_CONFIGS:
            self.current_region = region_code
            self.region_config = REGION_CONFIGS[region_code]
            self.logger.info(f"Region set to {region_code}")
        else:
            self.logger.warning(f"Region {region_code} not configured, using {self.default_region}")
            self.current_region = self.default_region
            self.region_config = REGION_CONFIGS[self.default_region]
    
    def get_text(self, key: str, **kwargs) -> str:
        """
        Get localized text for a key
        
        Args:
            key: Translation key (e.g., 'api.separation.success')
            **kwargs: Variables to substitute in the text
            
        Returns:
            Localized text string
        """
        
        # Get translation for current language
        lang_translations = self.translations.get(self.current_language, {})
        
        # Fall back to default language if key not found
        if key not in lang_translations:
            lang_translations = self.translations.get(self.default_language, {})
        
        # Get the text
        text = lang_translations.get(key, key)  # Return key if not found
        
        # Substitute variables
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error formatting text for key {key}: {e}")
        
        return text
    
    def get_privacy_notice(self) -> str:
        """Get appropriate privacy notice for current region"""
        
        if self.region_config.gdpr_required:
            return self.get_text("privacy.gdpr_notice")
        elif self.region_config.ccpa_required:
            return self.get_text("privacy.ccpa_notice")
        else:
            return self.get_text("privacy.data_processing")
    
    def get_data_retention_notice(self) -> str:
        """Get data retention notice for current region"""
        
        return self.get_text(
            "privacy.data_retention",
            days=self.region_config.max_data_retention_days
        )
    
    def get_compliance_requirements(self) -> Dict[str, Any]:
        """Get compliance requirements for current region"""
        
        return {
            "region": self.current_region,
            "gdpr_required": self.region_config.gdpr_required,
            "ccpa_required": self.region_config.ccpa_required,
            "pdpa_required": self.region_config.pdpa_required,
            "data_residency_required": self.region_config.data_residency_required,
            "audit_logging_required": self.region_config.audit_logging_required,
            "encryption_required": self.region_config.encryption_required,
            "max_data_retention_days": self.region_config.max_data_retention_days,
            "additional_compliance": self.region_config.additional_compliance
        }
    
    def validate_data_processing(self, user_consent: bool = False) -> Dict[str, Any]:
        """
        Validate if data processing is allowed under current region's laws
        
        Args:
            user_consent: Whether user has given explicit consent
            
        Returns:
            Validation result with allowed status and requirements
        """
        
        result = {
            "allowed": True,
            "requirements": [],
            "warnings": []
        }
        
        # Check consent requirements
        if self.region_config.gdpr_required or self.region_config.ccpa_required:
            if not user_consent:
                result["allowed"] = False
                result["requirements"].append("explicit_consent")
        
        # Check data residency
        if self.region_config.data_residency_required:
            result["requirements"].append("data_residency")
            result["warnings"].append("Data must be processed within regional boundaries")
        
        # Check encryption
        if self.region_config.encryption_required:
            result["requirements"].append("encryption")
        
        # Check audit logging
        if self.region_config.audit_logging_required:
            result["requirements"].append("audit_logging")
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        
        return list(self.translations.keys())
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported region codes"""
        
        return list(REGION_CONFIGS.keys())
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable language name"""
        
        language_names = {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh-CN": "简体中文",
            "zh-TW": "繁體中文",
            "ko": "한국어",
            "pt": "Português",
            "it": "Italiano",
            "ru": "Русский",
            "ar": "العربية"
        }
        
        return language_names.get(language_code, language_code)


# Global localization manager instance
_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager() -> LocalizationManager:
    """Get global localization manager instance"""
    
    global _localization_manager
    
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    
    return _localization_manager


def initialize_localization(language: str = "en", region: str = "US") -> LocalizationManager:
    """Initialize global localization with specific language and region"""
    
    global _localization_manager
    
    _localization_manager = LocalizationManager(language, region)
    
    return _localization_manager


def get_text(key: str, **kwargs) -> str:
    """Convenience function to get localized text"""
    
    return get_localization_manager().get_text(key, **kwargs)


def set_language(language_code: str):
    """Convenience function to set language"""
    
    get_localization_manager().set_language(language_code)


def set_region(region_code: str):
    """Convenience function to set region"""
    
    get_localization_manager().set_region(region_code)


# Decorator for API endpoints to automatically handle localization
def localized_response(f):
    """Decorator to add localization context to API responses"""
    
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        
        # Add localization metadata to response if it's a dict
        if isinstance(result, dict):
            lm = get_localization_manager()
            result["_localization"] = {
                "language": lm.current_language,
                "region": lm.current_region,
                "compliance": lm.get_compliance_requirements()
            }
        
        return result
    
    return wrapper
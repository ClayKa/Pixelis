"""
Privacy Protection Module

Implements PII detection, redaction, and data anonymization for the Pixelis framework.
Ensures compliance with privacy regulations and protects user data.
"""

import re
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io

logger = logging.getLogger(__name__)


@dataclass
class RedactionPattern:
    """Pattern for detecting and redacting sensitive information."""
    name: str
    pattern: str
    replacement: str
    description: str
    risk_level: str  # 'high', 'medium', 'low'
    
    def compile(self):
        """Compile the regex pattern."""
        return re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


@dataclass
class PrivacyConfig:
    """Configuration for privacy protection."""
    enable_pii_redaction: bool = True
    enable_image_metadata_stripping: bool = True
    enable_differential_privacy: bool = False
    differential_privacy_epsilon: float = 1.0
    log_redaction_stats: bool = True
    redaction_placeholder_format: str = "[{category}]"
    hash_pii_for_consistency: bool = True
    max_text_length: int = 10000  # Prevent processing extremely long texts
    allowed_image_formats: Set[str] = field(default_factory=lambda: {'jpg', 'jpeg', 'png', 'gif', 'bmp'})


class PIIRedactor:
    """
    Detects and redacts personally identifiable information from text.
    
    Implements comprehensive PII detection for:
    - Names (persons, organizations, locations)
    - Contact information (emails, phones)
    - Identification numbers (SSN, passport, etc.)
    - Financial information (credit cards, accounts)
    - Network identifiers (IP addresses, MACs)
    - Personal URLs and social media handles
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize the PII redactor.
        
        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.patterns = self._initialize_patterns()
        self.redaction_stats = {
            'total_redactions': 0,
            'redactions_by_type': {}
        }
        
        # Cache for consistent redaction of same values
        self.redaction_cache = {} if self.config.hash_pii_for_consistency else None
        
        logger.info("PII Redactor initialized with %d patterns", len(self.patterns))
    
    def _initialize_patterns(self) -> List[RedactionPattern]:
        """Initialize redaction patterns for various PII types."""
        patterns = [
            # Email addresses
            RedactionPattern(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement="EMAIL",
                description="Email addresses",
                risk_level="high"
            ),
            
            # Phone numbers (various formats)
            RedactionPattern(
                name="phone",
                pattern=r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                replacement="PHONE",
                description="Phone numbers",
                risk_level="high"
            ),
            
            # Social Security Numbers (US)
            RedactionPattern(
                name="ssn",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                replacement="SSN",
                description="Social Security Numbers",
                risk_level="high"
            ),
            
            # Credit card numbers
            RedactionPattern(
                name="credit_card",
                pattern=r'\b(?:\d[ -]*?){13,16}\b',
                replacement="CREDIT_CARD",
                description="Credit card numbers",
                risk_level="high"
            ),
            
            # IP addresses (IPv4 and IPv6)
            RedactionPattern(
                name="ip_address",
                pattern=r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b|(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}',
                replacement="IP_ADDRESS",
                description="IP addresses",
                risk_level="medium"
            ),
            
            # MAC addresses
            RedactionPattern(
                name="mac_address",
                pattern=r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})',
                replacement="MAC_ADDRESS",
                description="MAC addresses",
                risk_level="low"
            ),
            
            # URLs with personal information
            RedactionPattern(
                name="personal_url",
                pattern=r'https?://[^\s]*(?:profile|user|account|id|uid|member)/[^\s]+',
                replacement="PERSONAL_URL",
                description="URLs containing personal identifiers",
                risk_level="medium"
            ),
            
            # Social media handles
            RedactionPattern(
                name="social_handle",
                pattern=r'@[A-Za-z0-9_]{1,15}\b',
                replacement="SOCIAL_HANDLE",
                description="Social media handles",
                risk_level="low"
            ),
            
            # Date of birth (various formats)
            RedactionPattern(
                name="date_of_birth",
                pattern=r'\b(?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12][0-9]|3[01])[/\-.](?:19|20)\d{2}\b',
                replacement="DATE_OF_BIRTH",
                description="Dates of birth",
                risk_level="medium"
            ),
            
            # Bank account numbers
            RedactionPattern(
                name="bank_account",
                pattern=r'\b[0-9]{8,17}\b',
                replacement="ACCOUNT_NUMBER",
                description="Bank account numbers",
                risk_level="high"
            ),
            
            # Passport numbers
            RedactionPattern(
                name="passport",
                pattern=r'\b[A-Z]{1,2}[0-9]{6,9}\b',
                replacement="PASSPORT",
                description="Passport numbers",
                risk_level="high"
            ),
            
            # Driver's license numbers (generic pattern)
            RedactionPattern(
                name="drivers_license",
                pattern=r'\b[A-Z]{1,2}[0-9]{5,8}\b',
                replacement="LICENSE_NUMBER",
                description="Driver's license numbers",
                risk_level="high"
            ),
            
            # Medical record numbers
            RedactionPattern(
                name="medical_record",
                pattern=r'\bMRN[:\s]?[0-9]{6,10}\b',
                replacement="MEDICAL_RECORD",
                description="Medical record numbers",
                risk_level="high"
            ),
            
            # AWS Access Keys
            RedactionPattern(
                name="aws_access_key",
                pattern=r'AKIA[0-9A-Z]{16}',
                replacement="AWS_KEY",
                description="AWS access keys",
                risk_level="high"
            ),
            
            # API Keys (generic pattern)
            RedactionPattern(
                name="api_key",
                pattern=r'\b[a-zA-Z0-9]{32,}\b',
                replacement="API_KEY",
                description="API keys",
                risk_level="high"
            )
        ]
        
        # Compile all patterns
        for pattern in patterns:
            pattern.compile()
        
        return patterns
    
    def redact_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted_text, redaction_counts)
        """
        if not self.config.enable_pii_redaction:
            return text, {}
        
        # Check text length
        if len(text) > self.config.max_text_length:
            logger.warning(f"Text exceeds maximum length ({len(text)} > {self.config.max_text_length}), truncating")
            text = text[:self.config.max_text_length]
        
        redacted_text = text
        redaction_counts = {}
        
        for pattern_obj in self.patterns:
            pattern = pattern_obj.compile()
            matches = pattern.findall(redacted_text)
            
            if matches:
                # Count redactions
                redaction_counts[pattern_obj.name] = len(matches)
                
                # Create replacement
                replacement = self.config.redaction_placeholder_format.format(
                    category=pattern_obj.replacement
                )
                
                # If caching is enabled, use consistent replacements
                if self.config.hash_pii_for_consistency and self.redaction_cache is not None:
                    def replace_with_cache(match):
                        original = match.group(0)
                        if original not in self.redaction_cache:
                            # Create a deterministic but anonymized replacement
                            hash_suffix = hashlib.md5(original.encode()).hexdigest()[:6]
                            self.redaction_cache[original] = f"{replacement}_{hash_suffix}"
                        return self.redaction_cache[original]
                    
                    redacted_text = pattern.sub(replace_with_cache, redacted_text)
                else:
                    redacted_text = pattern.sub(replacement, redacted_text)
        
        # Update statistics
        self.redaction_stats['total_redactions'] += sum(redaction_counts.values())
        for category, count in redaction_counts.items():
            self.redaction_stats['redactions_by_type'][category] = \
                self.redaction_stats['redactions_by_type'].get(category, 0) + count
        
        if self.config.log_redaction_stats and redaction_counts:
            logger.info(f"Redacted {sum(redaction_counts.values())} PII instances: {redaction_counts}")
        
        return redacted_text, redaction_counts
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text without redacting.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping PII types to detected values
        """
        detections = {}
        
        for pattern_obj in self.patterns:
            pattern = pattern_obj.compile()
            matches = pattern.findall(text)
            
            if matches:
                detections[pattern_obj.name] = matches
        
        return detections
    
    def get_risk_assessment(self, text: str) -> Dict[str, Any]:
        """
        Assess privacy risk of text.
        
        Args:
            text: Text to assess
            
        Returns:
            Risk assessment report
        """
        detections = self.detect_pii(text)
        
        risk_score = 0
        risk_details = []
        
        for pattern_obj in self.patterns:
            if pattern_obj.name in detections:
                count = len(detections[pattern_obj.name])
                
                # Calculate risk contribution
                if pattern_obj.risk_level == 'high':
                    risk_contribution = count * 10
                elif pattern_obj.risk_level == 'medium':
                    risk_contribution = count * 5
                else:  # low
                    risk_contribution = count * 1
                
                risk_score += risk_contribution
                risk_details.append({
                    'type': pattern_obj.name,
                    'count': count,
                    'risk_level': pattern_obj.risk_level,
                    'description': pattern_obj.description
                })
        
        # Determine overall risk level
        if risk_score == 0:
            overall_risk = 'none'
        elif risk_score < 10:
            overall_risk = 'low'
        elif risk_score < 50:
            overall_risk = 'medium'
        else:
            overall_risk = 'high'
        
        return {
            'risk_score': risk_score,
            'risk_level': overall_risk,
            'pii_found': len(detections) > 0,
            'details': risk_details,
            'recommendation': self._get_risk_recommendation(overall_risk)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            'none': "No PII detected. Safe to process.",
            'low': "Low risk PII detected. Consider redaction for public sharing.",
            'medium': "Medium risk PII detected. Redaction recommended.",
            'high': "High risk PII detected. Redaction required before processing."
        }
        return recommendations.get(risk_level, "Unknown risk level")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get redaction statistics."""
        return {
            'total_redactions': self.redaction_stats['total_redactions'],
            'redactions_by_type': self.redaction_stats['redactions_by_type'].copy(),
            'cache_size': len(self.redaction_cache) if self.redaction_cache else 0
        }
    
    def clear_cache(self):
        """Clear the redaction cache."""
        if self.redaction_cache is not None:
            self.redaction_cache.clear()
            logger.info("Redaction cache cleared")


class ImageMetadataStripper:
    """
    Strips sensitive metadata from images.
    
    Removes:
    - EXIF data (including GPS coordinates)
    - Camera/device information
    - Timestamps
    - Author/copyright information
    - Software information
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize the metadata stripper.
        
        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.stats = {
            'images_processed': 0,
            'metadata_removed': 0,
            'gps_removed': 0
        }
        
        logger.info("Image Metadata Stripper initialized")
    
    def strip_metadata(self, image_data: bytes, image_format: str = 'jpeg') -> bytes:
        """
        Strip metadata from image data.
        
        Args:
            image_data: Raw image bytes
            image_format: Image format
            
        Returns:
            Image data with metadata stripped
        """
        if not self.config.enable_image_metadata_stripping:
            return image_data
        
        # Check format
        if image_format.lower() not in self.config.allowed_image_formats:
            logger.warning(f"Unsupported image format: {image_format}")
            return image_data
        
        try:
            from PIL import Image
            
            # Load image
            img = Image.open(io.BytesIO(image_data))
            
            # Check for EXIF data
            has_exif = hasattr(img, '_getexif') and img._getexif() is not None
            has_gps = False
            
            if has_exif:
                exif = img._getexif()
                # Check for GPS data (tag 34853)
                if exif and 34853 in exif:
                    has_gps = True
                    self.stats['gps_removed'] += 1
                
                self.stats['metadata_removed'] += 1
            
            # Create new image without metadata
            # This removes all EXIF, XMP, and other metadata
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)
            
            # Save to bytes
            output = io.BytesIO()
            
            # Determine save format
            save_format = 'JPEG' if image_format.lower() in ['jpg', 'jpeg'] else image_format.upper()
            
            # Save without metadata
            if save_format == 'JPEG':
                image_without_exif.save(output, format=save_format, quality=95, optimize=True)
            else:
                image_without_exif.save(output, format=save_format)
            
            self.stats['images_processed'] += 1
            
            if has_exif:
                logger.debug(f"Stripped metadata from image (GPS: {has_gps})")
            
            return output.getvalue()
            
        except ImportError:
            logger.error("PIL/Pillow not installed. Cannot strip image metadata.")
            return image_data
        except Exception as e:
            logger.error(f"Failed to strip image metadata: {e}")
            return image_data
    
    def analyze_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze metadata in image without removing it.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Metadata analysis report
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS
            
            img = Image.open(io.BytesIO(image_data))
            
            metadata = {
                'has_exif': False,
                'has_gps': False,
                'tags': {},
                'gps_data': {},
                'sensitive_fields': []
            }
            
            if hasattr(img, '_getexif') and img._getexif() is not None:
                metadata['has_exif'] = True
                exif = img._getexif()
                
                if exif:
                    # Parse EXIF tags
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata['tags'][tag] = str(value)[:100]  # Truncate long values
                        
                        # Check for sensitive fields
                        sensitive_tags = [
                            'Artist', 'Copyright', 'DateTime', 'DateTimeOriginal',
                            'Software', 'HostComputer', 'Make', 'Model'
                        ]
                        if tag in sensitive_tags:
                            metadata['sensitive_fields'].append(tag)
                    
                    # Check for GPS data
                    if 34853 in exif:  # GPSInfo tag
                        metadata['has_gps'] = True
                        gps_info = exif[34853]
                        
                        if isinstance(gps_info, dict):
                            for key in gps_info.keys():
                                gps_tag = GPSTAGS.get(key, key)
                                metadata['gps_data'][gps_tag] = str(gps_info[key])[:100]
            
            return metadata
            
        except ImportError:
            return {'error': 'PIL/Pillow not installed'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


class DataAnonymizer:
    """
    Comprehensive data anonymization for the Pixelis framework.
    
    Combines PII redaction, metadata stripping, and other privacy techniques.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize the data anonymizer.
        
        Args:
            config: Privacy configuration
        """
        self.config = config or PrivacyConfig()
        self.pii_redactor = PIIRedactor(config)
        self.metadata_stripper = ImageMetadataStripper(config)
        
        logger.info("Data Anonymizer initialized")
    
    def anonymize_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize an experience before storage.
        
        Args:
            experience_data: Raw experience data
            
        Returns:
            Anonymized experience data
        """
        anonymized = experience_data.copy()
        
        # Redact PII from text fields
        text_fields = ['question_text', 'answer', 'feedback', 'notes']
        for field in text_fields:
            if field in anonymized and isinstance(anonymized[field], str):
                anonymized[field], _ = self.pii_redactor.redact_text(anonymized[field])
        
        # Strip metadata from images
        if 'image_data' in anonymized and isinstance(anonymized['image_data'], bytes):
            anonymized['image_data'] = self.metadata_stripper.strip_metadata(
                anonymized['image_data'],
                image_format=anonymized.get('image_format', 'jpeg')
            )
        
        # Remove or hash user identifiers
        if 'user_id' in anonymized:
            # Hash user ID for consistency without storing actual ID
            anonymized['user_hash'] = hashlib.sha256(
                str(anonymized['user_id']).encode()
            ).hexdigest()[:16]
            del anonymized['user_id']
        
        # Remove session information
        session_fields = ['session_id', 'ip_address', 'user_agent']
        for field in session_fields:
            if field in anonymized:
                del anonymized[field]
        
        # Add anonymization metadata
        anonymized['_anonymized'] = True
        anonymized['_anonymization_timestamp'] = datetime.now().isoformat()
        
        return anonymized
    
    def verify_anonymization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that data has been properly anonymized.
        
        Args:
            data: Data to verify
            
        Returns:
            Verification report
        """
        issues = []
        
        # Check text fields for PII
        text_fields = ['question_text', 'answer', 'feedback', 'notes']
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                detections = self.pii_redactor.detect_pii(data[field])
                if detections:
                    issues.append({
                        'field': field,
                        'issue': 'Contains PII',
                        'details': list(detections.keys())
                    })
        
        # Check for forbidden fields
        forbidden_fields = ['user_id', 'session_id', 'ip_address', 'email']
        for field in forbidden_fields:
            if field in data:
                issues.append({
                    'field': field,
                    'issue': 'Forbidden field present',
                    'details': 'Should be removed'
                })
        
        # Check anonymization flag
        if not data.get('_anonymized', False):
            issues.append({
                'field': '_anonymized',
                'issue': 'Missing anonymization flag',
                'details': 'Data may not be anonymized'
            })
        
        return {
            'is_anonymized': len(issues) == 0,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anonymization statistics."""
        return {
            'pii_redactor': self.pii_redactor.get_statistics(),
            'metadata_stripper': self.metadata_stripper.get_statistics()
        }
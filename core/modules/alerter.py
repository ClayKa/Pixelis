"""
Alerter Module

Provides real-time alerting capabilities for critical system events.
Supports multiple alert channels including logging, webhooks, and email.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import urllib.request
import urllib.error
from collections import deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'component': self.component,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AlertThreshold:
    """Configuration for an alert threshold."""
    metric_name: str
    threshold_value: float
    comparison: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    severity: AlertSeverity
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    description: str = ""
    
    def check(self, value: float) -> bool:
        """Check if the value exceeds the threshold."""
        if self.comparison == 'gt':
            return value > self.threshold_value
        elif self.comparison == 'lt':
            return value < self.threshold_value
        elif self.comparison == 'eq':
            return value == self.threshold_value
        elif self.comparison == 'gte':
            return value >= self.threshold_value
        elif self.comparison == 'lte':
            return value <= self.threshold_value
        else:
            logger.warning(f"Unknown comparison operator: {self.comparison}")
            return False


class Alerter:
    """
    Central alerting system for the Pixelis framework.
    
    Monitors system health metrics and sends alerts when thresholds are exceeded.
    Implements cooldown periods to prevent alert spam.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alerter.
        
        Args:
            config: Configuration dictionary containing alert settings
        """
        self.config = config
        
        # Alert thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Alert history for cooldown management
        self.alert_history: Dict[str, datetime] = {}
        self.alert_lock = threading.Lock()
        
        # Recent alerts buffer
        self.recent_alerts = deque(maxlen=100)
        
        # Alert channels
        self.channels = {
            'log': self._alert_to_log,
            'webhook': self._alert_to_webhook,
            'file': self._alert_to_file
        }
        
        # Enable/disable specific channels
        self.enabled_channels = config.get('alert_channels', ['log'])
        
        # Webhook configuration
        self.webhook_url = config.get('webhook_url')
        self.webhook_timeout = config.get('webhook_timeout', 5)
        
        # File alerting
        self.alert_file_path = config.get('alert_file_path', './alerts.jsonl')
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': {s.value: 0 for s in AlertSeverity},
            'alerts_by_component': {},
            'last_alert_time': None
        }
        
        logger.info(f"Alerter initialized with channels: {self.enabled_channels}")
    
    def _initialize_thresholds(self) -> List[AlertThreshold]:
        """Initialize alert thresholds from configuration."""
        thresholds = []
        
        # Default thresholds
        default_thresholds = [
            AlertThreshold(
                metric_name='update_rate',
                threshold_value=0.0,
                comparison='eq',
                severity=AlertSeverity.WARNING,
                description="Model update rate dropped to zero",
                cooldown_seconds=600
            ),
            AlertThreshold(
                metric_name='faiss_failure_rate',
                threshold_value=0.1,
                comparison='gt',
                severity=AlertSeverity.CRITICAL,
                description="High k-NN search failure rate",
                cooldown_seconds=300
            ),
            AlertThreshold(
                metric_name='mean_kl_divergence',
                threshold_value=self.config.get('kl_alert_threshold', 0.2),
                comparison='gt',
                severity=AlertSeverity.WARNING,
                description="High KL divergence detected",
                cooldown_seconds=300
            ),
            AlertThreshold(
                metric_name='queue_size',
                threshold_value=self.config.get('queue_alert_threshold', 900),
                comparison='gt',
                severity=AlertSeverity.WARNING,
                description="Queue size approaching limit",
                cooldown_seconds=180
            ),
            AlertThreshold(
                metric_name='memory_usage_ratio',
                threshold_value=0.9,
                comparison='gt',
                severity=AlertSeverity.CRITICAL,
                description="Memory usage exceeds 90%",
                cooldown_seconds=300
            ),
            AlertThreshold(
                metric_name='inference_latency_p99',
                threshold_value=2.0,
                comparison='gt',
                severity=AlertSeverity.WARNING,
                description="P99 latency exceeds 2 seconds",
                cooldown_seconds=600
            )
        ]
        
        # Add custom thresholds from config
        custom_thresholds = self.config.get('custom_thresholds', [])
        for custom in custom_thresholds:
            thresholds.append(AlertThreshold(**custom))
        
        # Override with defaults if no custom thresholds
        if not custom_thresholds:
            thresholds.extend(default_thresholds)
        
        return thresholds
    
    def check_metrics(self, metrics: Dict[str, float], component: str = "system"):
        """
        Check metrics against thresholds and send alerts if needed.
        
        Args:
            metrics: Dictionary of metric values
            component: Component name for alert attribution
        """
        for threshold in self.thresholds:
            if threshold.metric_name in metrics:
                value = metrics[threshold.metric_name]
                
                if threshold.check(value):
                    # Check cooldown
                    if not self._is_in_cooldown(threshold.metric_name, threshold.cooldown_seconds):
                        self.send_alert(
                            severity=threshold.severity,
                            component=component,
                            message=f"{threshold.description}: {threshold.metric_name}={value:.3f}",
                            details={
                                'metric': threshold.metric_name,
                                'value': value,
                                'threshold': threshold.threshold_value,
                                'comparison': threshold.comparison
                            }
                        )
    
    def send_alert(
        self,
        severity: AlertSeverity,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Send an alert through configured channels.
        
        Args:
            severity: Alert severity level
            component: Component that triggered the alert
            message: Alert message
            details: Additional alert details
        """
        import uuid
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            component=component,
            message=message,
            details=details or {}
        )
        
        # Update history and stats
        with self.alert_lock:
            self.alert_history[f"{component}:{message[:50]}"] = alert.timestamp
            self.recent_alerts.append(alert)
            
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_severity'][severity.value] += 1
            self.stats['alerts_by_component'][component] = \
                self.stats['alerts_by_component'].get(component, 0) + 1
            self.stats['last_alert_time'] = alert.timestamp
        
        # Send through enabled channels
        for channel in self.enabled_channels:
            if channel in self.channels:
                try:
                    self.channels[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _is_in_cooldown(self, key: str, cooldown_seconds: int) -> bool:
        """
        Check if an alert is in cooldown period.
        
        Args:
            key: Alert key for cooldown tracking
            cooldown_seconds: Cooldown period in seconds
            
        Returns:
            True if in cooldown, False otherwise
        """
        with self.alert_lock:
            if key in self.alert_history:
                last_alert_time = self.alert_history[key]
                time_since_alert = (datetime.now() - last_alert_time).total_seconds()
                return time_since_alert < cooldown_seconds
        return False
    
    def _alert_to_log(self, alert: Alert):
        """Send alert to log."""
        if alert.severity == AlertSeverity.INFO:
            logger.info(f"[ALERT] {alert.component}: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"[ALERT] {alert.component}: {alert.message}")
        elif alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            logger.critical(f"[ALERT] {alert.component}: {alert.message}")
    
    def _alert_to_webhook(self, alert: Alert):
        """Send alert to webhook (e.g., Slack, Discord)."""
        if not self.webhook_url:
            return
        
        try:
            # Prepare webhook payload
            payload = {
                'text': f"*{alert.severity.value.upper()}*: {alert.message}",
                'attachments': [{
                    'color': self._get_severity_color(alert.severity),
                    'fields': [
                        {'title': 'Component', 'value': alert.component, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp.isoformat(), 'short': True},
                        {'title': 'Details', 'value': json.dumps(alert.details, indent=2)}
                    ]
                }]
            }
            
            # Send webhook request
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=self.webhook_timeout) as response:
                if response.status != 200:
                    logger.warning(f"Webhook returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _alert_to_file(self, alert: Alert):
        """Append alert to file in JSONL format."""
        try:
            with open(self.alert_file_path, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get color code for severity level (for webhooks)."""
        colors = {
            AlertSeverity.INFO: '#36a64f',     # Green
            AlertSeverity.WARNING: '#ff9900',   # Orange
            AlertSeverity.CRITICAL: '#ff0000',  # Red
            AlertSeverity.EMERGENCY: '#800080'  # Purple
        }
        return colors.get(severity, '#808080')  # Gray default
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        with self.alert_lock:
            alerts = list(self.recent_alerts)
            return alerts[-limit:] if len(alerts) > limit else alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alerter statistics."""
        with self.alert_lock:
            return self.stats.copy()
    
    def clear_history(self):
        """Clear alert history (useful for testing)."""
        with self.alert_lock:
            self.alert_history.clear()
            self.recent_alerts.clear()
            logger.info("Alert history cleared")


class HealthMonitor:
    """
    System health monitoring with integrated alerting.
    
    Tracks key health indicators and triggers alerts when thresholds are exceeded.
    """
    
    def __init__(self, alerter: Alerter):
        """
        Initialize the health monitor.
        
        Args:
            alerter: Alerter instance for sending alerts
        """
        self.alerter = alerter
        
        # Health metrics
        self.metrics = {
            'update_rate': 0.0,
            'faiss_failure_rate': 0.0,
            'mean_kl_divergence': 0.0,
            'queue_sizes': {},
            'memory_usage_ratio': 0.0,
            'inference_latency_p99': 0.0
        }
        
        # Metric history for rate calculation
        self.update_history = deque(maxlen=60)  # Last 60 measurements
        self.faiss_attempts = 0
        self.faiss_failures = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def update_metrics(self, new_metrics: Dict[str, Any], component: str = "system"):
        """
        Update health metrics and check for alerts.
        
        Args:
            new_metrics: New metric values to update
            component: Component source of the metrics
        """
        with self.lock:
            # Update metrics
            for key, value in new_metrics.items():
                if key in self.metrics:
                    self.metrics[key] = value
                elif key == 'queue_sizes' and isinstance(value, dict):
                    self.metrics['queue_sizes'].update(value)
            
            # Calculate derived metrics
            self._calculate_rates()
            
            # Check thresholds and send alerts
            self.alerter.check_metrics(self.metrics, component)
    
    def record_update(self, timestamp: Optional[datetime] = None):
        """Record a model update for rate calculation."""
        with self.lock:
            self.update_history.append(timestamp or datetime.now())
            self._calculate_rates()
    
    def record_faiss_attempt(self, success: bool):
        """Record a FAISS operation attempt."""
        with self.lock:
            self.faiss_attempts += 1
            if not success:
                self.faiss_failures += 1
            
            # Calculate failure rate
            if self.faiss_attempts > 0:
                self.metrics['faiss_failure_rate'] = self.faiss_failures / self.faiss_attempts
    
    def _calculate_rates(self):
        """Calculate rate-based metrics."""
        # Calculate update rate (updates per minute)
        if len(self.update_history) > 1:
            time_span = (self.update_history[-1] - self.update_history[0]).total_seconds()
            if time_span > 0:
                self.metrics['update_rate'] = (len(self.update_history) - 1) / (time_span / 60.0)
        
        # Check individual queue sizes
        for queue_name, size in self.metrics.get('queue_sizes', {}).items():
            if size > 0:  # Only check non-empty queues
                self.alerter.check_metrics(
                    {'queue_size': size},
                    component=f"queue:{queue_name}"
                )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            return {
                'metrics': self.metrics.copy(),
                'is_healthy': self._is_system_healthy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _is_system_healthy(self) -> bool:
        """Determine if the system is healthy based on metrics."""
        # Define health criteria
        return (
            self.metrics['faiss_failure_rate'] < 0.1 and
            self.metrics['mean_kl_divergence'] < 0.3 and
            self.metrics['memory_usage_ratio'] < 0.95
        )
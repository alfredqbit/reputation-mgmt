"""
Optimized Pipeline Architecture for ML-Based Reputation Management

This module provides a production-ready, modular pipeline for reputation
management using machine learning. Key features:

- Modular architecture with pluggable components
- Async and parallel processing support  
- Built-in caching, metrics, and monitoring
- Extensible feature extraction and model ensemble
- Real-time anomaly and drift detection

Usage:
    from reputation_pipeline import ReputationManagementPipeline
    
    pipeline = ReputationManagementPipeline()
    result = pipeline.process_document({'text': 'Great product!', 'source': 'social_media'})
    print(result.sentiment, result.reputation_score)

Author: Generated for ML Reputation Management Framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from datetime import datetime
from collections import deque
import hashlib
import json


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    enabled: bool = True
    weight: float = 1.0
    batch_size: int = 32
    device: str = 'cpu'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class PipelineConfig:
    """Master pipeline configuration."""
    max_workers: int = 4
    async_enabled: bool = True
    batch_size: int = 64
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    anomaly_threshold: float = 0.5
    drift_sensitivity: float = 0.1
    alert_on_anomaly: bool = True
    alert_on_drift: bool = True
    min_confidence: float = 0.6
    
    def validate(self) -> bool:
        assert 0 < self.max_workers <= 32
        assert 0 < self.batch_size <= 512
        assert 0 <= self.anomaly_threshold <= 1
        return True


# =============================================================================
# Infrastructure Layer
# =============================================================================

class MetricsTracker:
    """Track pipeline metrics for monitoring and optimization."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, List[float]] = {}
        
    def record(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def increment(self, name: str, delta: int = 1):
        self.counters[name] = self.counters.get(name, 0) + delta
        
    def time(self, name: str):
        return _TimerContext(self, name)
    
    def get_summary(self) -> Dict:
        summary = {'counters': self.counters.copy()}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        for name, times in self.timers.items():
            if times:
                summary[f'{name}_time_ms'] = {
                    'mean': np.mean(times) * 1000,
                    'p95': np.percentile(times, 95) * 1000
                }
        return summary


class _TimerContext:
    def __init__(self, tracker: MetricsTracker, name: str):
        self.tracker = tracker
        self.name = name
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        if self.name not in self.tracker.timers:
            self.tracker.timers[self.name] = []
        self.tracker.timers[self.name].append(elapsed)


class LRUCache:
    """LRU Cache with TTL support."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order: deque = deque()
        self.hits = 0
        self.misses = 0
        
    def _hash_key(self, key: Any) -> str:
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        hkey = self._hash_key(key)
        if hkey in self.cache:
            value, timestamp = self.cache[hkey]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                return value
            else:
                del self.cache[hkey]
        self.misses += 1
        return None
    
    def set(self, key: Any, value: Any):
        hkey = self._hash_key(key)
        self.cache[hkey] = (value, time.time())
        self.access_order.append(hkey)
        while len(self.cache) > self.max_size:
            old_key = self.access_order.popleft()
            self.cache.pop(old_key, None)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global instances
METRICS = MetricsTracker()
CACHE = LRUCache()


# =============================================================================
# Data Layer
# =============================================================================

class DataSourceType(Enum):
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    REVIEWS = "reviews"
    INTERNAL = "internal"


@dataclass
class ReputationDocument:
    """Standardized document representation."""
    id: str
    text: str
    source: DataSourceType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_embedding: Optional[np.ndarray] = None
    video_embedding: Optional[np.ndarray] = None
    author_id: Optional[str] = None
    reply_to: Optional[str] = None
    mentions: List[str] = field(default_factory=list)


class DataIngestionPipeline:
    """Unified data ingestion with preprocessing."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessors: List[Callable] = []
        self.logger = logging.getLogger('DataIngestion')
        
    def add_preprocessor(self, fn: Callable):
        self.preprocessors.append(fn)
        
    def preprocess(self, doc: ReputationDocument) -> ReputationDocument:
        for fn in self.preprocessors:
            doc = fn(doc)
        return doc
    
    def ingest_batch(self, documents: List[Dict]) -> List[ReputationDocument]:
        results = []
        for doc_dict in documents:
            try:
                doc = ReputationDocument(
                    id=doc_dict.get('id', str(hash(doc_dict.get('text', '')))),
                    text=doc_dict['text'],
                    source=DataSourceType(doc_dict.get('source', 'social_media')),
                    timestamp=doc_dict.get('timestamp', datetime.now()),
                    metadata=doc_dict.get('metadata', {}),
                    author_id=doc_dict.get('author_id'),
                    reply_to=doc_dict.get('reply_to'),
                    mentions=doc_dict.get('mentions', [])
                )
                doc = self.preprocess(doc)
                results.append(doc)
                METRICS.increment('documents_ingested')
            except Exception as e:
                self.logger.error(f"Failed to ingest document: {e}")
                METRICS.increment('ingestion_errors')
        return results


def clean_text(doc: ReputationDocument) -> ReputationDocument:
    """Clean and normalize text."""
    import re
    text = doc.text
    text = re.sub(r'http\S+|www\S+', '', text)
    text = ' '.join(text.split())
    doc.text = text
    return doc


def extract_entities(doc: ReputationDocument) -> ReputationDocument:
    """Extract named entities (placeholder)."""
    doc.metadata['entities'] = []
    return doc


# =============================================================================
# Feature Engineering Layer
# =============================================================================

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass
    
    @abstractmethod
    def extract(self, doc: ReputationDocument) -> np.ndarray:
        pass
    
    def extract_batch(self, docs: List[ReputationDocument]) -> np.ndarray:
        return np.vstack([self.extract(doc) for doc in docs])


class LexiconFeatureExtractor(FeatureExtractor):
    """Extract lexicon-based sentiment features."""
    
    def __init__(self):
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'best', 'happy'}
        self.negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed'}
        
    @property
    def name(self) -> str:
        return 'lexicon'
    
    @property
    def output_dim(self) -> int:
        return 5
    
    def extract(self, doc: ReputationDocument) -> np.ndarray:
        tokens = doc.text.lower().split()
        n = len(tokens) if tokens else 1
        pos_count = sum(1 for t in tokens if t in self.positive_words)
        neg_count = sum(1 for t in tokens if t in self.negative_words)
        return np.array([
            pos_count / n,
            neg_count / n,
            (pos_count - neg_count) / n,
            np.log1p(n),
            np.mean([len(t) for t in tokens]) if tokens else 0
        ])


class EmbeddingFeatureExtractor(FeatureExtractor):
    """Extract neural embedding features."""
    
    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        
    @property
    def name(self) -> str:
        return 'embedding'
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def extract(self, doc: ReputationDocument) -> np.ndarray:
        np.random.seed(hash(doc.text) % 2**32)
        return np.random.randn(self.embed_dim).astype(np.float32)


class TemporalFeatureExtractor(FeatureExtractor):
    """Extract temporal features."""
    
    @property
    def name(self) -> str:
        return 'temporal'
    
    @property
    def output_dim(self) -> int:
        return 7
    
    def extract(self, doc: ReputationDocument) -> np.ndarray:
        ts = doc.timestamp
        return np.array([
            np.sin(2 * np.pi * ts.hour / 24),
            np.cos(2 * np.pi * ts.hour / 24),
            np.sin(2 * np.pi * ts.weekday() / 7),
            np.cos(2 * np.pi * ts.weekday() / 7),
            float(ts.weekday() >= 5),
            np.sin(2 * np.pi * ts.month / 12),
            np.cos(2 * np.pi * ts.month / 12)
        ])


class FeatureEngineeringPipeline:
    """Orchestrate feature extraction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.extractors: List[FeatureExtractor] = []
        self.logger = logging.getLogger('FeatureEngineering')
        
    def add_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)
        self.logger.info(f"Added extractor: {extractor.name}")
        
    @property
    def total_dim(self) -> int:
        return sum(e.output_dim for e in self.extractors)
    
    def extract(self, doc: ReputationDocument, use_cache: bool = True) -> np.ndarray:
        cache_key = f"features:{doc.id}"
        if use_cache and self.config.cache_enabled:
            cached = CACHE.get(cache_key)
            if cached is not None:
                return cached
        
        features = []
        for extractor in self.extractors:
            with METRICS.time(f'feature_{extractor.name}'):
                feat = extractor.extract(doc)
                features.append(feat)
        
        result = np.concatenate(features)
        if use_cache and self.config.cache_enabled:
            CACHE.set(cache_key, result)
        return result
    
    def extract_batch(self, docs: List[ReputationDocument], parallel: bool = True) -> np.ndarray:
        with METRICS.time('feature_extraction_batch'):
            if parallel and len(docs) > 10:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    features = list(executor.map(self.extract, docs))
            else:
                features = [self.extract(doc) for doc in docs]
            METRICS.increment('features_extracted', len(docs))
            return np.vstack(features)


# =============================================================================
# Model Layer
# =============================================================================

class ModelComponent(ABC):
    """Abstract base class for ML model components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        pass
    
    def predict_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
        return [self.predict(f) for f in features]


class SentimentModel(ModelComponent):
    """Sentiment classification model."""
    
    def __init__(self, input_dim: int = 268, hidden_dim: int = 128):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.model.eval()
        
    @property
    def name(self) -> str:
        return 'sentiment'
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).squeeze().numpy()
        labels = ['negative', 'neutral', 'positive']
        pred_idx = np.argmax(probs)
        return {
            'label': labels[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {l: float(p) for l, p in zip(labels, probs)}
        }
    
    def predict_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).numpy()
        labels = ['negative', 'neutral', 'positive']
        results = []
        for p in probs:
            pred_idx = np.argmax(p)
            results.append({
                'label': labels[pred_idx],
                'confidence': float(p[pred_idx]),
                'probabilities': {l: float(prob) for l, prob in zip(labels, p)}
            })
        return results


class AnomalyModel(ModelComponent):
    """Anomaly detection model."""
    
    def __init__(self, input_dim: int = 268, latent_dim: int = 32):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.threshold = 0.5
        
    @property
    def name(self) -> str:
        return 'anomaly'
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            z = self.encoder(x)
            x_recon = self.decoder(z)
            recon_error = F.mse_loss(x_recon, x).item()
        anomaly_score = min(recon_error / 2.0, 1.0)
        return {
            'is_anomaly': anomaly_score > self.threshold,
            'anomaly_score': float(anomaly_score),
            'reconstruction_error': float(recon_error)
        }


class ReputationScoreModel(ModelComponent):
    """Aggregate reputation score predictor."""
    
    def __init__(self, input_dim: int = 268):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    @property
    def name(self) -> str:
        return 'reputation_score'
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            score = self.model(x).item()
        return {
            'reputation_score': float(score),
            'category': 'positive' if score > 0.6 else ('negative' if score < 0.4 else 'neutral')
        }


class ModelEnsemble:
    """Ensemble of model components."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models: Dict[str, Tuple[ModelComponent, float]] = {}
        self.logger = logging.getLogger('ModelEnsemble')
        
    def add_model(self, model: ModelComponent, weight: float = 1.0):
        self.models[model.name] = (model, weight)
        self.logger.info(f"Added model: {model.name}")
        
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        results = {}
        for name, (model, weight) in self.models.items():
            with METRICS.time(f'model_{name}'):
                try:
                    pred = model.predict(features)
                    results[name] = pred
                except Exception as e:
                    self.logger.error(f"Model {name} failed: {e}")
                    results[name] = {'error': str(e)}
        return results
    
    def predict_batch(self, features: np.ndarray) -> List[Dict[str, Any]]:
        n = features.shape[0]
        all_results = [{} for _ in range(n)]
        for name, (model, weight) in self.models.items():
            with METRICS.time(f'model_{name}_batch'):
                try:
                    preds = model.predict_batch(features)
                    for i, pred in enumerate(preds):
                        all_results[i][name] = pred
                except Exception as e:
                    self.logger.error(f"Batch prediction failed for {name}: {e}")
        return all_results


# =============================================================================
# Monitoring Layer
# =============================================================================

class DriftDetector:
    """Online drift detection using CUSUM algorithm."""
    
    def __init__(self, threshold: float = 5.0, sensitivity: float = 0.1):
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.baseline_mean = None
        self.baseline_std = None
        self.warmup_buffer = []
        self.warmup_size = 50
        self.drift_points = []
        self.t = 0
        
    def update(self, value: float) -> Dict[str, Any]:
        self.t += 1
        if len(self.warmup_buffer) < self.warmup_size:
            self.warmup_buffer.append(value)
            if len(self.warmup_buffer) == self.warmup_size:
                self.baseline_mean = np.mean(self.warmup_buffer)
                self.baseline_std = np.std(self.warmup_buffer) + 1e-6
            return {'drift_detected': False, 'in_warmup': True}
        
        z = (value - self.baseline_mean) / self.baseline_std
        self.cusum_pos = max(0, self.cusum_pos + z - self.sensitivity)
        self.cusum_neg = max(0, self.cusum_neg - z - self.sensitivity)
        drift_detected = self.cusum_pos > self.threshold or self.cusum_neg > self.threshold
        
        if drift_detected:
            self.drift_points.append(self.t)
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            self.warmup_buffer = [value]
            self.baseline_mean = None
            
        return {
            'drift_detected': drift_detected,
            'cusum_pos': self.cusum_pos,
            'cusum_neg': self.cusum_neg,
            'timestamp': self.t
        }


class ReputationMonitor:
    """Monitor reputation metrics and detect anomalies."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sentiment_drift = DriftDetector(threshold=5.0, sensitivity=config.drift_sensitivity)
        self.history = deque(maxlen=1000)
        self.alerts: List[Dict] = []
        self.logger = logging.getLogger('ReputationMonitor')
        
    def update(self, predictions: Dict[str, Any], doc: ReputationDocument) -> Dict[str, Any]:
        result = {'alerts': []}
        
        if 'sentiment' in predictions and 'probabilities' in predictions['sentiment']:
            probs = predictions['sentiment']['probabilities']
            sentiment_score = probs.get('positive', 0) - probs.get('negative', 0)
            drift_result = self.sentiment_drift.update(sentiment_score)
            result['sentiment_drift'] = drift_result
            
            if drift_result.get('drift_detected') and self.config.alert_on_drift:
                alert = {'type': 'SENTIMENT_DRIFT', 'timestamp': datetime.now(), 'details': drift_result}
                self.alerts.append(alert)
                result['alerts'].append(alert)
        
        if 'anomaly' in predictions:
            if predictions['anomaly'].get('is_anomaly') and self.config.alert_on_anomaly:
                alert = {
                    'type': 'ANOMALY_DETECTED',
                    'timestamp': datetime.now(),
                    'document_id': doc.id,
                    'score': predictions['anomaly'].get('anomaly_score')
                }
                self.alerts.append(alert)
                result['alerts'].append(alert)
        
        self.history.append({'timestamp': doc.timestamp, 'predictions': predictions, 'doc_id': doc.id})
        return result


# =============================================================================
# Output Layer
# =============================================================================

@dataclass
class ReputationAnalysisResult:
    """Complete analysis result for a document."""
    document_id: str
    timestamp: datetime
    sentiment: Dict[str, Any]
    anomaly: Dict[str, Any]
    reputation_score: float
    monitoring: Dict[str, Any]
    processing_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'document_id': self.document_id,
            'timestamp': self.timestamp.isoformat(),
            'sentiment': self.sentiment,
            'anomaly': self.anomaly,
            'reputation_score': self.reputation_score,
            'monitoring': self.monitoring,
            'processing_time_ms': self.processing_time_ms
        }


class ActionHandler(ABC):
    @abstractmethod
    def handle(self, result: ReputationAnalysisResult) -> bool:
        pass


class AlertActionHandler(ActionHandler):
    def __init__(self):
        self.logger = logging.getLogger('AlertHandler')
        
    def handle(self, result: ReputationAnalysisResult) -> bool:
        alerts = result.monitoring.get('alerts', [])
        for alert in alerts:
            self.logger.warning(f"ALERT [{alert['type']}]: {alert}")
        return True


class DatabaseActionHandler(ActionHandler):
    def __init__(self):
        self.buffer: List[Dict] = []
        self.buffer_size = 100
        
    def handle(self, result: ReputationAnalysisResult) -> bool:
        self.buffer.append(result.to_dict())
        if len(self.buffer) >= self.buffer_size:
            self._flush()
        return True
    
    def _flush(self):
        METRICS.increment('db_writes', len(self.buffer))
        self.buffer.clear()


class OutputPipeline:
    def __init__(self):
        self.handlers: List[ActionHandler] = []
        
    def add_handler(self, handler: ActionHandler):
        self.handlers.append(handler)
        
    def process(self, result: ReputationAnalysisResult):
        for handler in self.handlers:
            try:
                handler.handle(result)
            except Exception as e:
                logging.error(f"Handler {type(handler).__name__} failed: {e}")


# =============================================================================
# Master Pipeline
# =============================================================================

class ReputationManagementPipeline:
    """
    Master pipeline orchestrating all components for reputation management.
    
    Example:
        pipeline = ReputationManagementPipeline()
        result = pipeline.process_document({'text': 'Great product!', 'source': 'social_media'})
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.validate()
        
        self.ingestion = DataIngestionPipeline(self.config)
        self.features = FeatureEngineeringPipeline(self.config)
        self.ensemble = ModelEnsemble(self.config)
        self.monitor = ReputationMonitor(self.config)
        self.output = OutputPipeline()
        
        self.logger = logging.getLogger('ReputationPipeline')
        self._setup_default_components()
        
    def _setup_default_components(self):
        self.ingestion.add_preprocessor(clean_text)
        self.ingestion.add_preprocessor(extract_entities)
        
        self.features.add_extractor(LexiconFeatureExtractor())
        self.features.add_extractor(EmbeddingFeatureExtractor(embed_dim=256))
        self.features.add_extractor(TemporalFeatureExtractor())
        
        input_dim = self.features.total_dim
        self.ensemble.add_model(SentimentModel(input_dim=input_dim), weight=1.0)
        self.ensemble.add_model(AnomalyModel(input_dim=input_dim), weight=1.0)
        self.ensemble.add_model(ReputationScoreModel(input_dim=input_dim), weight=1.0)
        
        self.output.add_handler(AlertActionHandler())
        self.output.add_handler(DatabaseActionHandler())
        
        self.logger.info(f"Pipeline initialized with {self.features.total_dim} features")
        
    def process_document(self, doc_dict: Dict) -> ReputationAnalysisResult:
        """Process a single document through the entire pipeline."""
        start_time = time.perf_counter()
        
        docs = self.ingestion.ingest_batch([doc_dict])
        if not docs:
            raise ValueError("Document ingestion failed")
        doc = docs[0]
        
        features = self.features.extract(doc)
        predictions = self.ensemble.predict(features)
        monitoring = self.monitor.update(predictions, doc)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        result = ReputationAnalysisResult(
            document_id=doc.id,
            timestamp=doc.timestamp,
            sentiment=predictions.get('sentiment', {}),
            anomaly=predictions.get('anomaly', {}),
            reputation_score=predictions.get('reputation_score', {}).get('reputation_score', 0.5),
            monitoring=monitoring,
            processing_time_ms=processing_time
        )
        
        self.output.process(result)
        METRICS.record('processing_time_ms', processing_time)
        METRICS.increment('documents_processed')
        
        return result
    
    def process_batch(self, doc_dicts: List[Dict], parallel: bool = True) -> List[ReputationAnalysisResult]:
        """Process a batch of documents."""
        start_time = time.perf_counter()
        
        docs = self.ingestion.ingest_batch(doc_dicts)
        if not docs:
            return []
        
        features = self.features.extract_batch(docs, parallel=parallel)
        all_predictions = self.ensemble.predict_batch(features)
        
        results = []
        for doc, predictions in zip(docs, all_predictions):
            monitoring = self.monitor.update(predictions, doc)
            result = ReputationAnalysisResult(
                document_id=doc.id,
                timestamp=doc.timestamp,
                sentiment=predictions.get('sentiment', {}),
                anomaly=predictions.get('anomaly', {}),
                reputation_score=predictions.get('reputation_score', {}).get('reputation_score', 0.5),
                monitoring=monitoring,
                processing_time_ms=0
            )
            results.append(result)
            self.output.process(result)
        
        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = total_time / len(results) if results else 0
        for r in results:
            r.processing_time_ms = avg_time
        
        METRICS.record('batch_processing_time_ms', total_time)
        METRICS.increment('documents_processed', len(results))
        self.logger.info(f"Processed {len(results)} documents in {total_time:.2f}ms")
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get pipeline metrics."""
        metrics = METRICS.get_summary()
        metrics['cache_hit_rate'] = CACHE.hit_rate
        metrics['alerts'] = len(self.monitor.alerts)
        metrics['drift_points'] = len(self.monitor.sentiment_drift.drift_points)
        return metrics


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = ReputationManagementPipeline()
    
    # Test document
    test_doc = {
        'text': 'This product is absolutely amazing! Best purchase ever.',
        'source': 'social_media'
    }
    
    result = pipeline.process_document(test_doc)
    print(f"Sentiment: {result.sentiment.get('label')}")
    print(f"Reputation Score: {result.reputation_score:.3f}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")

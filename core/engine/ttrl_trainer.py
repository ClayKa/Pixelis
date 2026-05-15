"""
TTRL online training backend.

This module wires the Pixelis online learning components into a concrete
training loop:

- model loading through a local HuggingFace checkpoint or a user supplied loader
- request stream ingestion from JSON/JSONL
- inference and confidence-gated update scheduling through InferenceEngine
- asynchronous model updates through UpdateWorker
- experience-buffer and checkpoint artifact reporting through TTRLContext

The implementation deliberately avoids mock predictions or synthetic rewards.
If no local model or request stream is configured, it fails before training.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.modules.reward_shaping import RewardOrchestrator
from core.modules.voting import TemporalEnsembleVoting
from core.reproducibility import ArtifactType

logger = logging.getLogger(__name__)


def _deep_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_callable(import_path: str) -> Callable[..., Any]:
    if ":" not in import_path:
        raise ValueError(
            f"Loader path must use 'module:function' syntax, got: {import_path}"
        )

    module_name, function_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    if not callable(loader):
        raise TypeError(f"Configured loader is not callable: {import_path}")
    return loader


def _to_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=dtype)
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype)
    return None


def _as_batched_long(value: Any) -> Optional[torch.Tensor]:
    tensor = _to_tensor(value, dtype=torch.long)
    if tensor is not None and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class TTRLModelAdapter(nn.Module):
    """
    Normalizes model calls for Pixelis TTRL.

    InferenceEngine calls ``model.forward(input_data_dict)``. UpdateWorker calls
    the same object as a normal training module with ``input_ids`` / ``labels``.
    This adapter supports both without forcing every model implementation to
    know about Pixelis request dictionaries.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_config = generation_config or {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self.predict(args[0])
        return self._call_model(**kwargs)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs = self._prepare_model_kwargs(input_data, include_labels=False)

        with torch.no_grad():
            outputs = self._call_model(**model_kwargs)

        logits = getattr(outputs, "logits", outputs if isinstance(outputs, torch.Tensor) else None)
        answer = input_data.get("answer") or input_data.get("target_answer") or ""
        confidence = input_data.get("confidence")

        if logits is not None and isinstance(logits, torch.Tensor):
            confidence = float(torch.softmax(logits[..., -1, :], dim=-1).max().item())

            if not answer and self.generation_config.get("decode_argmax", True):
                token_id = int(logits[..., -1, :].argmax(dim=-1).flatten()[0].item())
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                    answer = self.tokenizer.decode([token_id], skip_special_tokens=True)
                else:
                    answer = str(token_id)

        if confidence is None:
            confidence = 0.5

        return {
            "answer": answer,
            "confidence": float(confidence),
            "logits": logits.detach().cpu() if isinstance(logits, torch.Tensor) else None,
            "trajectory": input_data.get("trajectory", []),
        }

    def _prepare_model_kwargs(
        self,
        input_data: Dict[str, Any],
        include_labels: bool,
    ) -> Dict[str, Any]:
        input_ids = _as_batched_long(input_data.get("input_ids"))
        attention_mask = _as_batched_long(input_data.get("attention_mask"))

        if input_ids is None and self.tokenizer is not None and input_data.get("question"):
            encoded = self.tokenizer(input_data["question"], return_tensors="pt")
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")

        if input_ids is None:
            raise ValueError(
                "TTRL request must include input_ids, or configure a tokenizer and question text."
            )

        kwargs: Dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        if include_labels and input_data.get("labels") is not None:
            kwargs["labels"] = _as_batched_long(input_data.get("labels"))

        image_features = _to_tensor(input_data.get("image_features"))
        if image_features is not None:
            kwargs["images"] = image_features

        return kwargs

    def _call_model(self, **kwargs: Any) -> Any:
        device = self._model_device()
        kwargs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
            if value is not None
        }

        try:
            return self.model(**kwargs)
        except TypeError:
            filtered = self._filter_supported_kwargs(kwargs)
            return self.model(**filtered)

    def _filter_supported_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(self.model.forward)
        except (TypeError, ValueError):
            return kwargs

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs

        supported = set(signature.parameters)
        filtered = {key: value for key, value in kwargs.items() if key in supported}

        if "images" in kwargs and "images" not in supported and "pixel_values" in supported:
            filtered["pixel_values"] = kwargs["images"]

        return filtered

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


@dataclass
class TTRLArtifacts:
    model_path: Path
    metrics_path: Path
    experience_stats_path: Path


class TTRLRequestStream:
    """Read online TTRL requests from JSONL or JSON."""

    def __init__(self, request_path: Path, max_steps: Optional[int] = None):
        if not request_path.exists():
            raise FileNotFoundError(
                f"TTRL request stream not found: {request_path}. "
                "Create a JSONL/JSON stream with question/input_ids/labels fields."
            )
        self.request_path = request_path
        self.max_steps = max_steps

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        count = 0
        for item in self._iter_raw():
            yield self._normalize_item(item, count)
            count += 1
            if self.max_steps is not None and count >= self.max_steps:
                break

    def _iter_raw(self) -> Iterable[Dict[str, Any]]:
        if self.request_path.suffix == ".jsonl":
            with open(self.request_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        with open(self.request_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            data = data.get("requests", data.get("data", []))
        if not isinstance(data, list):
            raise ValueError("TTRL JSON request file must be a list or contain a requests list.")

        for item in data:
            yield item

    def _normalize_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"TTRL request #{index} is not a JSON object")

        normalized = dict(item)
        normalized.setdefault("request_id", normalized.get("id", f"ttrl_request_{index}"))
        if "question" not in normalized and "prompt" in normalized:
            normalized["question"] = normalized["prompt"]
        return normalized


class TTRLBackend:
    """Concrete TTRL online training backend."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ttrl_config = config.get("ttrl", {})
        self.online_config = config.get("online", {})
        self.output_dir = Path(
            self.ttrl_config.get("output_dir")
            or _deep_get(config, "training", "output_dir", default="./outputs/ttrl")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, context: Any) -> Tuple[str, Dict[str, Any]]:
        model, tokenizer, processor = self._load_model()
        adapter = TTRLModelAdapter(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            generation_config=self.ttrl_config.get("generation", {}),
        )

        engine = self._build_engine(adapter)
        request_stream = self._build_request_stream()
        start_worker = self.ttrl_config.get("start_update_worker", True)

        metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "updates_enqueued": 0,
            "start_time": time.time(),
        }

        try:
            if start_worker:
                engine.start_update_worker()
            else:
                logger.warning(
                    "TTRL update worker is disabled by config. Inference and experience "
                    "collection will run, but online model weights will not update."
                )

            for step, request in enumerate(request_stream, start=1):
                result, confidence, metadata = asyncio.run(engine.infer_and_adapt(request))

                success = result is not None and "error" not in metadata
                metrics["total_requests"] += 1
                metrics["successful_requests"] += int(success)
                metrics["failed_requests"] += int(not success)
                metrics["updates_enqueued"] = engine.stats.get("total_updates", 0)

                if hasattr(context, "log_experience"):
                    context.log_experience(
                        experience_id=request.get("request_id", f"step_{step}"),
                        input_data={"question": request.get("question", "")},
                        output_data=result,
                        metadata={
                            "confidence": confidence,
                            "success": success,
                            "update_path": metadata.get("update_path"),
                        },
                    )

                if metadata.get("update_path") == "automatic" and hasattr(context, "log_online_update"):
                    context.log_online_update(
                        experience_id=request.get("request_id", f"step_{step}"),
                        reward=float(confidence),
                        confidence=float(confidence),
                        metadata={"step": step},
                    )

                if self._should_snapshot(step) and hasattr(context, "log_experience_buffer"):
                    context.log_experience_buffer(engine.experience_buffer.get_statistics())

            # Ensure worker consumes queued updates and writes final stats/snapshot.
            engine.shutdown()

        except Exception:
            engine.shutdown()
            raise

        metrics["duration_seconds"] = time.time() - metrics["start_time"]
        metrics["updates_enqueued"] = engine.stats.get("total_updates", metrics["updates_enqueued"])
        artifacts = self._write_outputs(adapter, engine, metrics)

        if hasattr(context, "log_artifact"):
            context.log_artifact(
                name="ttrl_training_summary",
                type=ArtifactType.METRICS,
                data=metrics,
                metadata={"output_dir": str(self.output_dir)},
            )

        return str(artifacts.model_path), metrics

    def _build_engine(self, model: nn.Module) -> InferenceEngine:
        buffer = ExperienceBuffer(
            max_size=int(self.online_config.get("buffer_size", 10000)),
            embedding_dim=int(self.ttrl_config.get("embedding_dim", 768)),
            similarity_metric=self.online_config.get("similarity_metric", "cosine"),
            enable_persistence=bool(self.online_config.get("enable_persistence", True)),
            persistence_path=self.online_config.get(
                "persistence_path",
                str(self.output_dir / "experience_buffer"),
            ),
            enable_auto_pruning=bool(self.online_config.get("enable_auto_pruning", True)),
        )

        engine_config = self._engine_config()
        voting = TemporalEnsembleVoting(
            strategy=engine_config.get("voting_strategy", "weighted"),
            min_votes_required=int(engine_config.get("min_votes_required", 3)),
            confidence_threshold=float(engine_config.get("confidence_threshold", 0.5)),
        )
        reward = RewardOrchestrator(engine_config)

        return InferenceEngine(
            model=model,
            experience_buffer=buffer,
            voting_module=voting,
            reward_orchestrator=reward,
            config=engine_config,
        )

    def _engine_config(self) -> Dict[str, Any]:
        config = {
            **self.online_config,
            **self.ttrl_config.get("engine", {}),
        }
        config.setdefault("model_save_path", str(self.checkpoint_dir))
        config.setdefault("max_queue_size", self.online_config.get("update_queue_size", 100))
        config.setdefault("base_learning_rate", self.online_config.get("max_learning_rate", 1e-4))
        config.setdefault("confidence_threshold", self.online_config.get("confidence_threshold", 0.7))
        config.setdefault("min_learning_rate", self.online_config.get("min_learning_rate", 1e-6))
        config.setdefault("max_learning_rate", self.online_config.get("max_learning_rate", 1e-4))
        config.setdefault("k_neighbors", self.online_config.get("k_neighbors", 5))
        config.setdefault("voting_strategy", self.online_config.get("voting_strategy", "weighted"))
        config.setdefault("min_votes_required", self.online_config.get("min_votes_required", 3))
        return config

    def _build_request_stream(self) -> TTRLRequestStream:
        request_path = self.ttrl_config.get("request_path") or self.online_config.get("request_path")
        if not request_path:
            raise ValueError(
                "TTRL requires ttrl.request_path or online.request_path pointing to a JSONL/JSON request stream."
            )

        max_steps = self.ttrl_config.get("max_steps") or self.online_config.get("num_steps")
        return TTRLRequestStream(Path(request_path), max_steps=max_steps)

    def _load_model(self) -> Tuple[nn.Module, Optional[Any], Optional[Any]]:
        loader_path = self.ttrl_config.get("model_loader")
        if loader_path:
            loaded = _load_callable(loader_path)(self.config)
            if isinstance(loaded, tuple):
                model = loaded[0]
                tokenizer = loaded[1] if len(loaded) > 1 else None
                processor = loaded[2] if len(loaded) > 2 else None
            else:
                model, tokenizer, processor = loaded, None, None
            if not isinstance(model, nn.Module):
                raise TypeError("Configured TTRL model_loader must return a torch.nn.Module")
            return model, tokenizer, processor

        model_path = (
            self.ttrl_config.get("model_path")
            or _deep_get(self.config, "model", "adapter_path")
            or _deep_get(self.config, "model", "base_model_path")
        )
        if not model_path:
            raise ValueError(
                "TTRL requires ttrl.model_loader or a local ttrl.model_path/model.base_model_path."
            )

        path = Path(model_path)
        allow_remote = bool(self.ttrl_config.get("allow_remote_download", False))
        if not path.exists() and not allow_remote:
            raise FileNotFoundError(
                f"TTRL model path does not exist locally: {model_path}. "
                "Set ttrl.allow_remote_download=true only when networked model downloads are intended."
            )

        try:
            import transformers
        except ImportError as exc:
            raise ImportError("TTRL HuggingFace loading requires transformers.") from exc

        auto_class_name = self.ttrl_config.get("auto_model_class", "AutoModelForCausalLM")
        model_cls = getattr(transformers, auto_class_name)

        load_kwargs = dict(self.ttrl_config.get("model_load_kwargs", {}))
        load_kwargs.setdefault("local_files_only", not allow_remote)
        load_kwargs.setdefault("trust_remote_code", bool(self.ttrl_config.get("trust_remote_code", False)))

        model = model_cls.from_pretrained(str(model_path), **load_kwargs)
        tokenizer = self._load_tokenizer(model_path, allow_remote)
        processor = self._load_processor(model_path, allow_remote)

        adapter_path = _deep_get(self.config, "model", "adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Loading a LoRA adapter requires peft.") from exc
            model = PeftModel.from_pretrained(model, adapter_path)

        return model, tokenizer, processor

    def _load_tokenizer(self, model_path: str, allow_remote: bool) -> Optional[Any]:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=not allow_remote,
                trust_remote_code=bool(self.ttrl_config.get("trust_remote_code", False)),
            )
        except Exception as exc:
            logger.warning("Tokenizer could not be loaded for TTRL: %s", exc)
            return None

    def _load_processor(self, model_path: str, allow_remote: bool) -> Optional[Any]:
        try:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                model_path,
                local_files_only=not allow_remote,
                trust_remote_code=bool(self.ttrl_config.get("trust_remote_code", False)),
            )
        except Exception:
            return None

    def _should_snapshot(self, step: int) -> bool:
        interval = int(self.ttrl_config.get("snapshot_interval_steps", 0) or 0)
        return interval > 0 and step % interval == 0

    def _write_outputs(
        self,
        model: TTRLModelAdapter,
        engine: InferenceEngine,
        metrics: Dict[str, Any],
    ) -> TTRLArtifacts:
        metrics_path = self.output_dir / "ttrl_metrics.json"
        experience_stats_path = self.output_dir / "experience_buffer_stats.json"

        latest_worker_snapshot = self._latest_worker_snapshot()
        if latest_worker_snapshot is not None:
            model_path = latest_worker_snapshot
        else:
            model_path = self.output_dir / "ttrl_model_final.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                    "note": "No worker EMA snapshot was produced; saved current model state.",
                },
                model_path,
            )

        metrics.update(
            {
                "final_model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "experience_stats_path": str(experience_stats_path),
            }
        )

        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)

        with open(experience_stats_path, "w", encoding="utf-8") as handle:
            json.dump(engine.experience_buffer.get_statistics(), handle, indent=2, default=str)

        return TTRLArtifacts(
            model_path=model_path,
            metrics_path=metrics_path,
            experience_stats_path=experience_stats_path,
        )

    def _latest_worker_snapshot(self) -> Optional[Path]:
        pointer = self.checkpoint_dir / "latest_model_version.txt"
        if pointer.exists():
            first_line = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
            candidate = self.checkpoint_dir / first_line
            if candidate.exists():
                return candidate

        snapshots = sorted(self.checkpoint_dir.glob("ema_model_snapshot.v*.pt"))
        return snapshots[-1] if snapshots else None

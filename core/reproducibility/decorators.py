"""
Decorators for automatic artifact tracking and reproducibility.
"""

import functools
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .artifact_manager import ArtifactManager, ArtifactType
from .experiment_context import ExperimentContext
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def track_artifacts(
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    capture_args: bool = True,
    capture_source: bool = False,
) -> Callable:
    """
    Decorator to automatically track function inputs and outputs as artifacts.
    
    Args:
        inputs: List of input artifact names to track
        outputs: List of output artifact types to track
        capture_args: Whether to capture function arguments
        capture_source: Whether to capture function source code
    
    Example:
        @track_artifacts(inputs=["dataset:v1"], outputs=["model", "metrics"])
        def train_model(config):
            # Training logic
            return model, metrics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get artifact manager
            manager = ArtifactManager()
            
            # Capture function metadata
            func_metadata = {
                "function": func.__name__,
                "module": func.__module__,
                "timestamp": time.time(),
            }
            
            # Capture arguments if requested
            if capture_args:
                try:
                    # Get function signature
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    
                    # Serialize arguments
                    func_metadata["arguments"] = {
                        k: _serialize_arg(v) for k, v in bound.arguments.items()
                    }
                except Exception as e:
                    logger.debug(f"Could not capture function arguments: {e}")
            
            # Capture source code if requested
            if capture_source:
                try:
                    func_metadata["source"] = inspect.getsource(func)
                    func_metadata["source_hash"] = hashlib.sha256(
                        func_metadata["source"].encode()
                    ).hexdigest()[:16]
                except Exception as e:
                    logger.debug(f"Could not capture function source: {e}")
            
            # Track input artifacts
            input_artifacts = []
            if inputs:
                for input_spec in inputs:
                    try:
                        # Parse input specification (e.g., "dataset:v1")
                        if ":" in input_spec:
                            name, version = input_spec.split(":", 1)
                        else:
                            name, version = input_spec, None
                        
                        # Use the artifact
                        artifact = manager.use_artifact(name, version)
                        input_artifacts.append(f"{artifact.name}:{artifact.version}")
                        
                        logger.info(f"Using input artifact: {artifact.name}:{artifact.version}")
                    except Exception as e:
                        logger.warning(f"Could not track input artifact {input_spec}: {e}")
            
            # Log function execution start
            if manager.run:
                start_artifact = manager.log_artifact(
                    name=f"function_start_{func.__name__}",
                    type=ArtifactType.METRICS,
                    data=func_metadata,
                    parent_artifacts=input_artifacts,
                )
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                success = True
                error = None
            except Exception as e:
                execution_time = time.time() - start_time
                success = False
                error = str(e)
                raise
            finally:
                # Log execution completion
                if manager.run:
                    completion_data = {
                        **func_metadata,
                        "execution_time": execution_time,
                        "success": success,
                        "error": error,
                    }
                    
                    completion_artifact = manager.log_artifact(
                        name=f"function_complete_{func.__name__}",
                        type=ArtifactType.METRICS,
                        data=completion_data,
                        parent_artifacts=[f"{start_artifact.name}:{start_artifact.version}"] if 'start_artifact' in locals() else [],
                    )
            
            # Track output artifacts
            if outputs and success:
                # Handle different return types
                if isinstance(result, tuple):
                    results = result
                elif isinstance(result, dict):
                    results = list(result.values())
                else:
                    results = [result]
                
                for i, (output_type, output_value) in enumerate(zip(outputs, results)):
                    try:
                        # Determine artifact type
                        if output_type == "model":
                            # Handle model checkpoints
                            if isinstance(output_value, (str, Path)):
                                manager.log_large_artifact(
                                    name=f"{func.__name__}_model",
                                    file_path=output_value,
                                    type=ArtifactType.MODEL,
                                    metadata={"function": func.__name__},
                                )
                            else:
                                logger.warning(f"Cannot track model output of type {type(output_value)}")
                        
                        elif output_type == "metrics":
                            # Handle metrics
                            manager.log_artifact(
                                name=f"{func.__name__}_metrics",
                                type=ArtifactType.METRICS,
                                data=output_value,
                                metadata={"function": func.__name__},
                            )
                        
                        elif output_type == "dataset":
                            # Handle datasets
                            if isinstance(output_value, (str, Path)):
                                manager.log_large_artifact(
                                    name=f"{func.__name__}_dataset",
                                    file_path=output_value,
                                    type=ArtifactType.DATASET,
                                    metadata={"function": func.__name__},
                                )
                            else:
                                manager.log_artifact(
                                    name=f"{func.__name__}_dataset",
                                    type=ArtifactType.DATASET,
                                    data=output_value,
                                    metadata={"function": func.__name__},
                                )
                        
                        else:
                            # Generic artifact
                            manager.log_artifact(
                                name=f"{func.__name__}_{output_type}",
                                type=ArtifactType.METRICS,
                                data=_serialize_arg(output_value),
                                metadata={"function": func.__name__},
                            )
                        
                        logger.info(f"Tracked output artifact: {output_type}")
                        
                    except Exception as e:
                        logger.warning(f"Could not track output artifact {output_type}: {e}")
            
            return result
        
        return wrapper
    
    return decorator


def reproducible(
    name: Optional[str] = None,
    capture_level: int = 2,
    offline_mode: Optional[bool] = None,
) -> Callable:
    """
    Decorator to make a function fully reproducible with automatic context management.
    
    Args:
        name: Experiment name (auto-generated if not provided)
        capture_level: Environment capture level (1=basic, 2=standard, 3=complete)
        offline_mode: Force offline mode (None = auto-detect)
    
    Example:
        @reproducible(name="train_experiment")
        def train_model(config):
            # Training logic
            return model
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from arguments
            config = None
            
            # Try to find config in args/kwargs
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Look for common config parameter names
            for param_name in ["config", "cfg", "conf", "args"]:
                if param_name in bound.arguments:
                    config = bound.arguments[param_name]
                    break
            
            # Create experiment name
            exp_name = name or f"{func.__name__}_{int(time.time())}"
            
            # Create experiment context
            from .config_capture import EnvironmentCaptureLevel
            capture_enum = EnvironmentCaptureLevel(capture_level)
            
            with ExperimentContext(
                config=config,
                name=exp_name,
                capture_level=capture_enum,
                offline_mode=offline_mode,
            ) as ctx:
                # Log function information
                ctx.log_artifact(
                    name="function_info",
                    type=ArtifactType.CODE,
                    data={
                        "function": func.__name__,
                        "module": func.__module__,
                        "source": inspect.getsource(func) if capture_level >= 2 else None,
                        "arguments": {k: _serialize_arg(v) for k, v in bound.arguments.items()},
                    },
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log result summary
                ctx.log_artifact(
                    name="function_result",
                    type=ArtifactType.METRICS,
                    data={
                        "result_type": type(result).__name__,
                        "result_summary": _serialize_arg(result, max_depth=2),
                    },
                )
                
                return result
        
        return wrapper
    
    return decorator


def checkpoint(
    frequency: Union[int, str] = "epoch",
    save_best: bool = True,
    metric: str = "loss",
    mode: str = "min",
) -> Callable:
    """
    Decorator to automatically checkpoint model during training.
    
    Args:
        frequency: Checkpoint frequency ("epoch", "step", or integer)
        save_best: Whether to save best checkpoint
        metric: Metric to monitor for best checkpoint
        mode: "min" or "max" for metric comparison
    
    Example:
        @checkpoint(frequency="epoch", save_best=True, metric="val_loss", mode="min")
        def train_epoch(model, data_loader, optimizer):
            # Training logic
            return {"loss": loss, "val_loss": val_loss}
    """
    def decorator(func: Callable) -> Callable:
        # Track best metric
        best_metric = float("inf") if mode == "min" else float("-inf")
        call_count = 0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal best_metric, call_count
            
            # Execute function
            result = func(*args, **kwargs)
            call_count += 1
            
            # Determine if we should checkpoint
            should_checkpoint = False
            
            if frequency == "epoch":
                should_checkpoint = True
            elif frequency == "step":
                should_checkpoint = True
            elif isinstance(frequency, int):
                should_checkpoint = (call_count % frequency == 0)
            
            if should_checkpoint:
                # Get artifact manager
                manager = ArtifactManager()
                
                if not manager.run:
                    logger.warning("No active run, skipping checkpoint")
                    return result
                
                # Extract model from arguments
                model = None
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                
                # Look for model in arguments
                for param_name in ["model", "net", "network"]:
                    if param_name in bound.arguments:
                        model = bound.arguments[param_name]
                        break
                
                if model is None:
                    logger.warning("Could not find model in function arguments")
                    return result
                
                # Extract metrics from result
                metrics = {}
                if isinstance(result, dict):
                    metrics = result
                elif hasattr(result, "__dict__"):
                    metrics = result.__dict__
                
                # Create checkpoint
                checkpoint_data = {
                    "call_count": call_count,
                    "metrics": metrics,
                    "timestamp": time.time(),
                }
                
                # Save regular checkpoint
                checkpoint_path = Path(f"checkpoints/{func.__name__}_step_{call_count}.pt")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    import torch
                    torch.save(model.state_dict(), checkpoint_path)
                    
                    # Log checkpoint artifact
                    manager.log_large_artifact(
                        name=f"checkpoint_step_{call_count}",
                        file_path=checkpoint_path,
                        type=ArtifactType.CHECKPOINT,
                        metadata=checkpoint_data,
                    )
                    
                    logger.info(f"Saved checkpoint at step {call_count}")
                    
                except Exception as e:
                    logger.error(f"Could not save checkpoint: {e}")
                
                # Save best checkpoint if applicable
                if save_best and metric in metrics:
                    current_metric = metrics[metric]
                    is_better = False
                    
                    if mode == "min":
                        is_better = current_metric < best_metric
                    else:
                        is_better = current_metric > best_metric
                    
                    if is_better:
                        best_metric = current_metric
                        
                        # Save best checkpoint
                        best_path = Path(f"checkpoints/{func.__name__}_best.pt")
                        
                        try:
                            import torch
                            torch.save(model.state_dict(), best_path)
                            
                            # Log best checkpoint artifact
                            manager.log_large_artifact(
                                name="checkpoint_best",
                                file_path=best_path,
                                type=ArtifactType.CHECKPOINT,
                                metadata={
                                    **checkpoint_data,
                                    "best_metric": best_metric,
                                    "metric_name": metric,
                                },
                            )
                            
                            logger.info(
                                f"Saved best checkpoint "
                                f"({metric}={best_metric:.4f})"
                            )
                            
                        except Exception as e:
                            logger.error(f"Could not save best checkpoint: {e}")
            
            return result
        
        return wrapper
    
    return decorator


def _serialize_arg(arg: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
    """
    Serialize function argument for logging.
    
    Args:
        arg: Argument to serialize
        max_depth: Maximum nesting depth
        current_depth: Current nesting depth
    
    Returns:
        Serializable representation of argument
    """
    if current_depth >= max_depth:
        return str(type(arg))
    
    # Handle basic types
    if isinstance(arg, (str, int, float, bool, type(None))):
        return arg
    
    # Handle paths
    if isinstance(arg, Path):
        return str(arg)
    
    # Handle lists/tuples
    if isinstance(arg, (list, tuple)):
        if len(arg) > 10:
            # Truncate long lists
            return [
                _serialize_arg(item, max_depth, current_depth + 1)
                for item in arg[:5]
            ] + ["..."] + [
                _serialize_arg(item, max_depth, current_depth + 1)
                for item in arg[-2:]
            ]
        return [
            _serialize_arg(item, max_depth, current_depth + 1)
            for item in arg
        ]
    
    # Handle dictionaries
    if isinstance(arg, dict):
        if len(arg) > 10:
            # Truncate large dicts
            items = list(arg.items())[:5]
            return {
                k: _serialize_arg(v, max_depth, current_depth + 1)
                for k, v in items
            }
        return {
            k: _serialize_arg(v, max_depth, current_depth + 1)
            for k, v in arg.items()
        }
    
    # Handle numpy arrays
    try:
        import numpy as np
        if isinstance(arg, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": arg.shape,
                "dtype": str(arg.dtype),
            }
    except ImportError:
        pass
    
    # Handle torch tensors
    try:
        import torch
        if isinstance(arg, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(arg.shape),
                "dtype": str(arg.dtype),
                "device": str(arg.device),
            }
    except ImportError:
        pass
    
    # Handle objects with __dict__
    if hasattr(arg, "__dict__") and not inspect.isclass(arg):
        return {
            "type": type(arg).__name__,
            "attributes": _serialize_arg(arg.__dict__, max_depth, current_depth + 1),
        }
    
    # Fallback to string representation
    return str(arg)
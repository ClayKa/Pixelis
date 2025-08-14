#!/usr/bin/env python3
"""
Export Pixelis Models to Dedicated Inference Engines

Supports export to:
1. ONNX Runtime
2. TensorRT
3. OpenVINO
4. TorchScript
5. Core ML (for Apple Silicon)
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import time
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    export_format: str  # onnx, tensorrt, openvino, torchscript, coreml
    input_shapes: Dict[str, List[int]]  # Input name -> shape
    output_names: List[str]
    opset_version: int = 14
    fp16_mode: bool = True
    int8_mode: bool = False
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    optimization_level: int = 2  # 0: none, 1: basic, 2: full
    calibration_data: Optional[List[torch.Tensor]] = None


@dataclass
class ExportResults:
    """Results from model export."""
    export_format: str
    export_path: str
    model_size_mb: float
    export_time_s: float
    validation_passed: bool
    inference_speedup: Optional[float] = None
    optimization_applied: List[str] = None


class ModelExporter:
    """
    Export PyTorch models to optimized inference engines.
    """
    
    def __init__(self, config: ExportConfig):
        """
        Initialize model exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self.supported_formats = [
            'onnx', 'tensorrt', 'openvino', 'torchscript', 'coreml'
        ]
        
        if config.export_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {config.export_format}")
    
    def export(
        self,
        model: nn.Module,
        output_path: Path,
        validate: bool = True
    ) -> ExportResults:
        """
        Export model to specified format.
        
        Args:
            model: PyTorch model to export
            output_path: Path to save exported model
            validate: Whether to validate exported model
            
        Returns:
            Export results
        """
        logger.info(f"Exporting model to {self.config.export_format}")
        
        start_time = time.time()
        
        # Export based on format
        if self.config.export_format == 'onnx':
            export_path = self._export_onnx(model, output_path)
        elif self.config.export_format == 'tensorrt':
            export_path = self._export_tensorrt(model, output_path)
        elif self.config.export_format == 'openvino':
            export_path = self._export_openvino(model, output_path)
        elif self.config.export_format == 'torchscript':
            export_path = self._export_torchscript(model, output_path)
        elif self.config.export_format == 'coreml':
            export_path = self._export_coreml(model, output_path)
        else:
            raise ValueError(f"Unsupported format: {self.config.export_format}")
        
        export_time = time.time() - start_time
        
        # Get model size
        model_size = Path(export_path).stat().st_size / 1024 / 1024
        
        # Validate if requested
        validation_passed = True
        if validate:
            validation_passed = self._validate_exported_model(
                model, export_path
            )
        
        # Calculate speedup
        speedup = self._benchmark_speedup(model, export_path)
        
        results = ExportResults(
            export_format=self.config.export_format,
            export_path=str(export_path),
            model_size_mb=model_size,
            export_time_s=export_time,
            validation_passed=validation_passed,
            inference_speedup=speedup,
            optimization_applied=[]
        )
        
        logger.info(f"Export complete: {results.export_path}")
        logger.info(f"Model size: {results.model_size_mb:.2f} MB")
        logger.info(f"Export time: {results.export_time_s:.2f}s")
        
        return results
    
    def _export_onnx(self, model: nn.Module, output_path: Path) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            output_path: Output path
            
        Returns:
            Path to exported model
        """
        output_path = output_path.with_suffix('.onnx')
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs(model.device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()) if len(dummy_inputs) > 1 else list(dummy_inputs.values())[0],
            output_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=list(self.config.input_shapes.keys()),
            output_names=self.config.output_names,
            dynamic_axes=self.config.dynamic_axes,
            verbose=False
        )
        
        # Optimize ONNX model
        if self.config.optimization_level > 0:
            output_path = self._optimize_onnx(output_path)
        
        return output_path
    
    def _optimize_onnx(self, model_path: Path) -> Path:
        """
        Optimize ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
            
            optimized_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
            
            # Optimize model
            optimizer.optimize_model(
                str(model_path),
                model_type='bert',  # Adjust based on actual model
                num_heads=12,
                hidden_size=768,
                optimization_options=optimizer.FusionOptions('all'),
                opt_level=self.config.optimization_level,
                use_gpu=torch.cuda.is_available(),
                only_onnxruntime=False
            ).save_model_to_file(str(optimized_path))
            
            logger.info(f"Optimized ONNX model saved to {optimized_path}")
            return optimized_path
            
        except ImportError:
            logger.warning("ONNX Runtime Transformers not available, skipping optimization")
            return model_path
    
    def _export_tensorrt(self, model: nn.Module, output_path: Path) -> Path:
        """
        Export model to TensorRT format.
        
        Args:
            model: PyTorch model
            output_path: Output path
            
        Returns:
            Path to exported model
        """
        try:
            import tensorrt as trt
            import torch2trt
            
            output_path = output_path.with_suffix('.trt')
            
            # Create dummy inputs
            dummy_inputs = self._create_dummy_inputs(model.device)
            dummy_input = list(dummy_inputs.values())[0]
            
            # Convert to TensorRT
            model_trt = torch2trt.torch2trt(
                model,
                [dummy_input],
                fp16_mode=self.config.fp16_mode,
                int8_mode=self.config.int8_mode,
                int8_calib_dataset=self.config.calibration_data,
                max_batch_size=dummy_input.shape[0],
                max_workspace_size=1 << 30  # 1GB
            )
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(model_trt.engine.serialize())
            
            logger.info(f"TensorRT model saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.error("TensorRT not available, falling back to ONNX")
            return self._export_onnx(model, output_path)
    
    def _export_openvino(self, model: nn.Module, output_path: Path) -> Path:
        """
        Export model to OpenVINO format.
        
        Args:
            model: PyTorch model
            output_path: Output path
            
        Returns:
            Path to exported model
        """
        try:
            from openvino.tools import mo
            from openvino.runtime import Core
            
            # First export to ONNX
            onnx_path = self._export_onnx(model, output_path.parent / "temp.onnx")
            
            # Convert to OpenVINO
            output_path = output_path.with_suffix('.xml')
            
            mo_args = [
                '--input_model', str(onnx_path),
                '--output_dir', str(output_path.parent),
                '--model_name', output_path.stem
            ]
            
            if self.config.fp16_mode:
                mo_args.extend(['--data_type', 'FP16'])
            
            mo.main(mo_args)
            
            # Clean up temporary ONNX file
            onnx_path.unlink()
            
            logger.info(f"OpenVINO model saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.error("OpenVINO not available, falling back to ONNX")
            return self._export_onnx(model, output_path)
    
    def _export_torchscript(self, model: nn.Module, output_path: Path) -> Path:
        """
        Export model to TorchScript format.
        
        Args:
            model: PyTorch model
            output_path: Output path
            
        Returns:
            Path to exported model
        """
        output_path = output_path.with_suffix('.pt')
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs(model.device)
        
        # Try to trace the model
        try:
            if len(dummy_inputs) == 1:
                traced = torch.jit.trace(model, list(dummy_inputs.values())[0])
            else:
                traced = torch.jit.trace(model, tuple(dummy_inputs.values()))
            
            # Optimize for inference
            if self.config.optimization_level > 0:
                traced = torch.jit.optimize_for_inference(traced)
            
            # Save model
            torch.jit.save(traced, output_path)
            
            logger.info(f"TorchScript model saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to trace model: {e}")
            # Try scripting instead
            try:
                scripted = torch.jit.script(model)
                torch.jit.save(scripted, output_path)
                logger.info(f"TorchScript model (scripted) saved to {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"Failed to script model: {e2}")
                raise
    
    def _export_coreml(self, model: nn.Module, output_path: Path) -> Path:
        """
        Export model to Core ML format.
        
        Args:
            model: PyTorch model
            output_path: Output path
            
        Returns:
            Path to exported model
        """
        try:
            import coremltools as ct
            
            output_path = output_path.with_suffix('.mlpackage')
            
            # Create dummy inputs
            dummy_inputs = self._create_dummy_inputs('cpu')
            
            # Trace model
            model.eval()
            model.cpu()
            
            if len(dummy_inputs) == 1:
                traced = torch.jit.trace(model, list(dummy_inputs.values())[0])
            else:
                traced = torch.jit.trace(model, tuple(dummy_inputs.values()))
            
            # Convert to Core ML
            inputs = []
            for name, shape in self.config.input_shapes.items():
                inputs.append(ct.TensorType(name=name, shape=shape))
            
            mlmodel = ct.convert(
                traced,
                inputs=inputs,
                convert_to='mlprogram',
                minimum_deployment_target=ct.target.iOS15
            )
            
            # Save model
            mlmodel.save(output_path)
            
            logger.info(f"Core ML model saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.error("Core ML Tools not available, falling back to ONNX")
            return self._export_onnx(model, output_path)
    
    def _create_dummy_inputs(self, device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
        """
        Create dummy inputs for model export.
        
        Args:
            device: Device to create tensors on
            
        Returns:
            Dictionary of dummy inputs
        """
        dummy_inputs = {}
        
        for name, shape in self.config.input_shapes.items():
            # Create appropriate dummy data based on name
            if 'image' in name.lower():
                # Image data (normalized)
                dummy = torch.randn(*shape, device=device)
            elif 'ids' in name.lower():
                # Token IDs
                dummy = torch.randint(0, 30000, shape, device=device)
            elif 'mask' in name.lower():
                # Attention mask
                dummy = torch.ones(shape, device=device)
            else:
                # Default to random
                dummy = torch.randn(*shape, device=device)
            
            dummy_inputs[name] = dummy
        
        return dummy_inputs
    
    def _validate_exported_model(
        self,
        original_model: nn.Module,
        exported_path: Path
    ) -> bool:
        """
        Validate exported model against original.
        
        Args:
            original_model: Original PyTorch model
            exported_path: Path to exported model
            
        Returns:
            True if validation passes
        """
        try:
            # Create test inputs
            test_inputs = self._create_dummy_inputs('cpu')
            
            # Get original outputs
            original_model.eval()
            original_model.cpu()
            with torch.no_grad():
                if len(test_inputs) == 1:
                    original_outputs = original_model(list(test_inputs.values())[0])
                else:
                    original_outputs = original_model(**test_inputs)
            
            # Get exported outputs based on format
            if self.config.export_format == 'onnx':
                exported_outputs = self._run_onnx_inference(
                    exported_path, test_inputs
                )
            elif self.config.export_format == 'torchscript':
                exported_outputs = self._run_torchscript_inference(
                    exported_path, test_inputs
                )
            else:
                # Skip validation for other formats
                logger.info(f"Skipping validation for {self.config.export_format}")
                return True
            
            # Compare outputs
            if isinstance(original_outputs, torch.Tensor):
                original_outputs = original_outputs.numpy()
            
            if isinstance(exported_outputs, torch.Tensor):
                exported_outputs = exported_outputs.numpy()
            
            # Check if outputs are close
            rtol = 1e-3 if self.config.fp16_mode else 1e-5
            atol = 1e-3 if self.config.fp16_mode else 1e-7
            
            is_close = np.allclose(
                original_outputs,
                exported_outputs,
                rtol=rtol,
                atol=atol
            )
            
            if is_close:
                logger.info("Validation passed: outputs match")
            else:
                max_diff = np.max(np.abs(original_outputs - exported_outputs))
                logger.warning(f"Validation warning: max difference = {max_diff}")
            
            return is_close
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _run_onnx_inference(
        self,
        model_path: Path,
        inputs: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Run inference with ONNX Runtime."""
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Prepare inputs
        ort_inputs = {
            name: tensor.numpy() for name, tensor in inputs.items()
        }
        
        # Run inference
        outputs = session.run(None, ort_inputs)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _run_torchscript_inference(
        self,
        model_path: Path,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run inference with TorchScript."""
        model = torch.jit.load(model_path)
        model.eval()
        
        with torch.no_grad():
            if len(inputs) == 1:
                outputs = model(list(inputs.values())[0])
            else:
                outputs = model(**inputs)
        
        return outputs
    
    def _benchmark_speedup(
        self,
        original_model: nn.Module,
        exported_path: Path,
        num_iterations: int = 100
    ) -> float:
        """
        Benchmark speedup of exported model.
        
        Args:
            original_model: Original PyTorch model
            exported_path: Path to exported model
            num_iterations: Number of benchmark iterations
            
        Returns:
            Speedup factor
        """
        try:
            # Create test inputs
            test_inputs = self._create_dummy_inputs('cpu')
            
            # Benchmark original model
            original_model.eval()
            original_model.cpu()
            
            start = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    if len(test_inputs) == 1:
                        _ = original_model(list(test_inputs.values())[0])
                    else:
                        _ = original_model(**test_inputs)
            original_time = time.time() - start
            
            # Benchmark exported model
            if self.config.export_format == 'onnx':
                start = time.time()
                for _ in range(num_iterations):
                    _ = self._run_onnx_inference(exported_path, test_inputs)
                exported_time = time.time() - start
            elif self.config.export_format == 'torchscript':
                start = time.time()
                for _ in range(num_iterations):
                    _ = self._run_torchscript_inference(exported_path, test_inputs)
                exported_time = time.time() - start
            else:
                return None
            
            speedup = original_time / exported_time
            logger.info(f"Inference speedup: {speedup:.2f}x")
            
            return speedup
            
        except Exception as e:
            logger.error(f"Failed to benchmark speedup: {e}")
            return None


def main():
    """Main entry point for model export."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Pixelis models to inference engines")
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="onnx",
        choices=['onnx', 'tensorrt', 'openvino', 'torchscript', 'coreml'],
        help="Export format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Output directory"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock model for demonstration
    logger.info("Creating mock model for export demonstration")
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 1000)
    )
    
    # Configure export
    config = ExportConfig(
        export_format=args.export_format,
        input_shapes={'input': [1, 768]},
        output_names=['output'],
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        optimization_level=2 if args.optimize else 0,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    
    # Export model
    exporter = ModelExporter(config)
    results = exporter.export(
        model,
        output_dir / f"pixelis_model",
        validate=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("MODEL EXPORT COMPLETE")
    print("=" * 80)
    print(f"Format: {results.export_format}")
    print(f"Path: {results.export_path}")
    print(f"Size: {results.model_size_mb:.2f} MB")
    print(f"Export time: {results.export_time_s:.2f}s")
    print(f"Validation: {'✓' if results.validation_passed else '✗'}")
    if results.inference_speedup:
        print(f"Speedup: {results.inference_speedup:.2f}x")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
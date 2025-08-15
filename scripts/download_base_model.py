import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Configuration ---
# Add your new model to the registry.
# The keys are the short, user-friendly names for the command line.
MODEL_REGISTRY = {
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

def download_model(model_name: str, models_dir: Path):
    """
    Downloads a specified model from the Hugging Face Hub to a local directory.

    Args:
        model_name (str): The short name of the model to download (must be in MODEL_REGISTRY).
        models_dir (Path): The root directory where models will be stored.
    """
    if model_name not in MODEL_REGISTRY:
        print(f"Error: Model '{model_name}' not found in the registry.")
        print(f"Available models are: {list(MODEL_REGISTRY.keys())}")
        return

    hf_repo_id = MODEL_REGISTRY[model_name]
    # Use the repo name's folder-friendly version as the local directory name
    # e.g., "Qwen/Qwen3-8B" -> "Qwen--Qwen3-8B"
    local_model_dir = models_dir / hf_repo_id.replace("/", "--")

    print("=" * 80)
    print(f"Starting download for model: '{model_name}'")
    print(f"Hugging Face Repository: {hf_repo_id}")
    print(f"Target local directory: {local_model_dir}")
    print("=" * 80)

    if local_model_dir.exists() and any(local_model_dir.iterdir()):
        print("\nModel directory already exists and is not empty. Skipping download.")
        print("If you want to re-download, please delete the directory first.")
    else:
        print("\nDownloading model... This may take a while depending on your network speed.")
        
        # Optional: Set proxies if you are behind a firewall
        # os.environ['HTTP_PROXY'] = "http://your-proxy-address:port"
        # os.environ['HTTPS_PROXY'] = "http://your-proxy-address:port"

        try:
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(local_model_dir),
                local_dir_use_symlinks=False,  # Recommended for Windows compatibility and simplicity
                resume_download=True,
            )
            print("\n✅ Model downloaded successfully!")
        except Exception as e:
            print(f"\n❌ An error occurred during download: {e}")
            print("Please check your network connection, proxy settings, and Hugging Face token.")
            return

    print("\n--- Next Steps ---")
    print(f"To use this model, ensure your configuration YAML file points to the correct local path.")
    print("Example `model_path`:")
    print(f"{local_model_dir.resolve()}")
    print("-" * 20)


if __name__ == "__main__":
    # The script is executed from the root of the project (e.g., `python scripts/download_base_model.py`)
    project_root = Path(__file__).parent.parent
    default_models_dir = project_root / "pretrained_models"

    parser = argparse.ArgumentParser(
        description="Download a base model from Hugging Face Hub for the Pixelis project."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl-7b-instruct", # Keep the original as default
        choices=list(MODEL_REGISTRY.keys()),
        help="The short name of the model to download.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_models_dir,
        help="The directory to save the pretrained models.",
    )

    args = parser.parse_args()
    
    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    download_model(args.model, args.output_dir)
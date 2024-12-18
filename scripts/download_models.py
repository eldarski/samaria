#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
import argparse

MODELS = {
    "config": {
        "repo_id": "rinna/japanese-clip-vit-b-16",
        "filename": "config.json",
        "path": "models/config.json",
        "required": True,
    },
    "vision_model": {
        "path": "models/clip_vision.pt",
        "required": True,
    },
    "text_model": {
        "path": "models/clip_text.pt",
        "required": True,
    },
}


def download_file(repo_id: str, filename: str, path: str, desc: str = None):
    """Download a file with progress bar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"File already exists: {path}")
        return

    print(f"Downloading {desc or filename} from {repo_id}")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=".cache"
        )
        # Copy to destination
        import shutil

        shutil.copy2(downloaded_path, path)
        print(f"Downloaded to {path}")
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}")
        print(
            f"Available files can be found at: https://huggingface.co/{repo_id}/tree/main"
        )
        raise


def list_available_files(repo_id: str):
    """List all available files in the repository."""
    try:
        files = list_repo_files(repo_id)
        print(f"\nAvailable files in {repo_id}:")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error listing files: {e}")


def export_model():
    """Export model to TorchScript format."""
    print("Exporting model to TorchScript...")
    script_dir = Path(__file__).parent
    export_script = script_dir / "export_model.py"

    # Check if models already exist
    if Path("models/clip_vision.pt").exists() and Path("models/clip_text.pt").exists():
        print("Models already exist, skipping export")
        return

    result = os.system(f"python {export_script}")
    if result != 0:
        raise RuntimeError("Failed to export model")
    print("Model exported successfully")


def main():
    parser = argparse.ArgumentParser(description="Download model files")
    parser.add_argument(
        "--models-dir", default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--list-files", action="store_true", help="List available files in repositories"
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip model export")
    parser.add_argument(
        "--force", action="store_true", help="Force model export even if files exist"
    )
    args = parser.parse_args()

    if args.list_files:
        for info in MODELS.values():
            list_available_files(info["repo_id"])
        return

    base_dir = Path(args.models_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for name, info in MODELS.items():
        if "repo_id" not in info:
            continue
        path = base_dir / Path(info["path"]).name
        download_file(info["repo_id"], info["filename"], path, name)

    if not args.skip_export or args.force:
        try:
            export_model()
        except Exception as e:
            print(f"Error exporting model: {e}")
            print("Please install required packages: pip install transformers torch")
            sys.exit(1)


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Union, List
import numpy as np
import torch


def ensure_path(path: Union[str, Path]) -> Path:
    """Convert string path to Path and ensure it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def batch_tensors(tensors: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
    """Split list of tensors into batches."""
    return [
        torch.stack(tensors[i : i + batch_size])
        for i in range(0, len(tensors), batch_size)
    ]


def normalize_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize embeddings to unit length."""
    if isinstance(embeddings, np.ndarray):
        return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

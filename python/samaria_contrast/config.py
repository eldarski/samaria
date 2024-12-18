from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SamariaConfig:
    """Configuration for Samaria models."""

    vision_model_path: str = "models/mobilenet_v2.pt"
    text_model_path: str = "models/text_encoder.pt"
    embeddings_dir: Path = Path("data/embeddings")
    use_cuda: bool = False
    image_size: int = 224
    embedding_dim: int = 256
    language: str = "ja"

    @classmethod
    def from_pretrained(cls, name: str = "base") -> "SamariaConfig":
        """Load a predefined configuration."""
        configs = {
            "base": cls(),
            "small": cls(embedding_dim=128),
            "tiny": cls(embedding_dim=64),
        }
        if name not in configs:
            raise ValueError(
                f"Unknown config name: {name}. Available: {list(configs.keys())}"
            )
        return configs[name]


class Config:
    def __init__(self):
        # Model paths
        self.model_dir: Path = Path("models")
        self.tokenizer_dir: Path = self.model_dir / "tokenizer"

        # Training parameters
        self.batch_size: int = 32
        self.learning_rate: float = 0.001
        self.num_epochs: int = 100
        self.embedding_dim: int = 512

        # Device configuration
        self.device: str = "cuda"
        self.num_workers: int = 4

        # Logging
        self.log_level: str = "info"
        self.log_dir: Optional[Path] = None

try:
    from . import _samaria_bindings as _lib
except ImportError as e:
    raise ImportError(
        "Failed to import C++ bindings. Make sure the package is installed correctly."
    ) from e

from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
from .config import SamariaConfig


class ContrastiveLearningSystem:
    """A system for Japanese text and image contrastive learning."""

    def __init__(self, config: Optional[SamariaConfig] = None):
        """Initialize the contrastive learning system."""
        if config is None:
            config = SamariaConfig()

        self._text_processor = _lib.JapaneseTextProcessor()
        self._image_processor = _lib.ImageProcessor()
        self._model = _lib.ContrastiveModel(config.embedding_dim)
        self._config = config

    def process_text(self, text: str) -> List[str]:
        """Process Japanese text into tokens."""
        # Japanese text should already be unicode string
        return self._text_processor.tokenize(text)

    def process_text_to_embedding(self, text: str) -> np.ndarray:
        """Process Japanese text into embeddings."""
        embeddings = np.array(
            self._text_processor.generate_embeddings(text), dtype=np.float32
        )
        self._save_text_embeddings(text, embeddings)
        return embeddings

    def process_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Process an image and extract features."""
        print(f"\n=== Processing image in Python ===")
        image_path = str(Path(image_path).resolve())
        print(f"Resolved path: {image_path}")

        print("Calling C++ extract_features...")
        features = self._image_processor.extract_features(image_path)
        print(f"Got features from C++: {type(features)}")

        # Keep reference to original tensor
        self._last_features = features

        print("Converting to numpy array...")
        features_np = np.array(features, dtype=np.float32)
        print(
            f"Converted to numpy: shape={features_np.shape}, dtype={features_np.dtype}"
        )

        # Keep reference to numpy array
        self._last_features_np = features_np

        print("Returning numpy array...")
        self._save_image_embeddings(image_path, features_np)
        return features_np

    def train(
        self,
        text_image_pairs: List[Tuple[str, str]],
        epochs: int = 10,
        batch_size: Optional[int] = None,
    ) -> None:
        """Train the model on text-image pairs.

        Args:
            text_image_pairs: List of (text, image_path) pairs
            epochs: Number of training epochs
            batch_size: Batch size for training (not implemented yet)
        """
        # Convert all image paths to absolute paths
        processed_pairs = [
            (text, str(Path(img_path).resolve())) for text, img_path in text_image_pairs
        ]
        self._model.train(processed_pairs, epochs)

    def compute_similarity(self, text: str, image_path: str) -> Tuple[float, str]:
        """Compute similarity between text and image."""
        print("\n=== Computing similarity ===")
        text_embedding = self.process_text_to_embedding(text)
        print(f"Text embedding shape: {text_embedding.shape}")

        image_embedding = self.process_image(image_path)
        print(f"Image embedding shape: {image_embedding.shape}")

        # Get image description
        description = self._image_processor.get_description(image_path)
        print(f"Image content: {description}")
        print(f"Input text: {text}")

        print("Calling C++ compute_similarity...")
        similarity = self._model.compute_similarity(text_embedding, image_embedding)
        print(f"Got similarity: {similarity}")

        return similarity, description

    def find_best_matches(
        self, query_text: str, image_paths: List[Union[str, Path]], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find the best matching images for a given text query.

        Args:
            query_text: Input Japanese text
            image_paths: List of paths to image files
            top_k: Number of top matches to return

        Returns:
            List of (image_path, similarity_score) tuples, sorted by similarity
        """
        scores = []
        for img_path in image_paths:
            img_path = str(Path(img_path).resolve())
            similarity, _ = self.compute_similarity(query_text, img_path)
            scores.append((img_path, similarity))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def process_batch(
        self, texts: List[str], images: List[Union[str, Path]], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process batches of texts and images efficiently."""
        text_embeddings = []
        image_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_images = images[i : i + batch_size]

            # Process text batch
            text_batch = [
                self._text_processor.generate_embeddings(t) for t in batch_texts
            ]
            text_embeddings.extend(text_batch)

            # Process image batch
            image_batch = [self.process_image(img) for img in batch_images]
            image_embeddings.extend(image_batch)

        return np.array(text_embeddings), np.array(image_embeddings)

    def describe_image(self, image_path: Union[str, Path]) -> str:
        """Get a description of the image content.

        Args:
            image_path: Path to the image file

        Returns:
            str: Description of the image content
        """
        print(f"\n=== Describing image ===")
        image_path = str(Path(image_path).resolve())
        print(f"Processing image: {image_path}")

        # Process image and get description
        features = self.process_image(image_path)
        description = self._image_processor.get_description(image_path)
        print(f"Detected content: {description}")

        return description

    def _save_text_embeddings(self, text: str, embeddings: np.ndarray) -> None:
        """Save text embeddings to disk."""
        save_dir = self._config.embeddings_dir / "text"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use hash of text as filename to avoid filesystem issues
        filename = f"{hash(text)}.npy"
        np.save(save_dir / filename, embeddings)

    def _save_image_embeddings(self, image_path: str, embeddings: np.ndarray) -> None:
        """Save image embeddings to disk."""
        save_dir = self._config.embeddings_dir / "image"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use image filename as base
        filename = Path(image_path).stem + "_emb.npy"
        np.save(save_dir / filename, embeddings)

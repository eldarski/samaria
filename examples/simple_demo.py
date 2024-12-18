from samaria_contrast import ContrastiveLearningSystem, SamariaConfig
from pathlib import Path
import numpy as np
import locale


def main():
    # Initialize with config
    config = SamariaConfig.from_pretrained("base")
    cls = ContrastiveLearningSystem(config)

    # Ensure proper encoding for Japanese text
    locale.setlocale(locale.LC_ALL, "ja_JP.UTF-8")

    # Create directories for embeddings
    embeddings_dir = Path("data/embeddings")
    text_emb_dir = embeddings_dir / "text"
    image_emb_dir = embeddings_dir / "image"

    for dir_path in [text_emb_dir, image_emb_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Test pairs of Japanese text and images
    test_pairs = [
        # Matching pairs
        ("赤い服の少女", "1.jpeg"),  # Girl in red clothes
        ("戦闘的な少女", "1.jpeg"),  # Combat-ready girl
        ("黒い髪の少女", "1.jpeg"),  # Girl with black hair
        ("炎のある少女", "1.jpeg"),  # Girl with flames
        ("アニメ風の少女", "1.jpeg"),  # Anime-style girl
        # Mismatched pairs for testing
        ("静かな���", "1.jpeg"),  # Peaceful scenery (mismatch)
        ("可愛い猫", "1.jpeg"),  # Cute cat (mismatch)
        ("青い空", "1.jpeg"),  # Blue sky (mismatch)
        ("美しい花", "1.jpeg"),  # Beautiful flower (mismatch)
        ("夕暮れの街", "1.jpeg"),  # Evening city (mismatch)
    ]

    # Process each pair
    for text, image_name in test_pairs:
        # Process text
        tokens = cls.process_text(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")

        # Generate and save text embedding
        text_embedding = cls.process_text_to_embedding(text)
        np.save(
            text_emb_dir / f"{image_name.replace('.jpeg', '_text.npy')}", text_embedding
        )

        # Process image and save embedding
        image_path = Path("data/images") / image_name
        print(f"Loading image from: {image_path.absolute()}")
        image_embedding = cls.process_image(image_path)
        print(f"Successfully loaded image")
        np.save(
            image_emb_dir / f"{image_name.replace('.jpeg', '_image.npy')}",
            image_embedding,
        )
        print(f"Image features shape: {image_embedding.shape}")

        # Calculate similarity
        similarity, description = cls.compute_similarity(text, str(image_path))
        print(f"Image content: {description}")
        print(f"Input text: {text}")
        print(f"Similarity score: {similarity:.3f}")

        print(f"Embeddings saved to: {cls._config.embeddings_dir}")
        print("-" * 80)  # Add separator line

    # Process text and image
    text = "赤い服の少女"  # "Girl in red clothes"
    image_path = "data/images/1.jpeg"

    # Get image description
    detected_content = cls.describe_image(image_path)
    print(f"Detected in image: {detected_content}")

    # Compare with provided text
    similarity, description = cls.compute_similarity(text, image_path)
    print(f"Similarity score: {similarity:.3f}")


if __name__ == "__main__":
    main()

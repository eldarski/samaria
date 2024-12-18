# Samaria

Japanese text and image contrastive learning library using CLIP.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/samaria.git
cd samaria
```

2. Build and run with Docker (recommended):

```bash
docker-compose up --build
```

## Quick Usage Example

```python
from samaria_contrast import ContrastiveLearningSystem

# Initialize
cls = ContrastiveLearningSystem()

# Test similarity between text and image
text = "赤い服の少女"  # Girl in red clothes
image_path = "data/images/1.jpeg"
similarity = cls.compute_similarity(text, image_path)
print(f"Similarity: {similarity:.3f}")
```

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- pybind11
- CMake

## License

[Your chosen license]

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

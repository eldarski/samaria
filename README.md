# Text and image contrastive learning Python library implemented in C++

## Installation

1. Clone the repository:

```bash
git clone https://github.com/eldarski/samaria.git
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

You can also run

```
python examples/simple_demo.py
```

## Dependencies

- Python 3.9
- PyTorch
- OpenCV
- pybind11
- CMake

## Model used

https://huggingface.co/rinna/japanese-clip-vit-b-16



-----------------------------
https://exim.dev

from pathlib import Path

# Model URLs
BERT_MODEL_URL = (
    "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/pytorch_model.bin"
)
RESNET_MODEL_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

# Cache directories
CACHE_DIR = Path.home() / ".cache" / "samaria"
MODEL_CACHE_DIR = CACHE_DIR / "models"
TOKENIZER_CACHE_DIR = CACHE_DIR / "tokenizer"

# Image processing
IMAGE_SIZE = 224
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# Training
MAX_SEQ_LENGTH = 512
TEMPERATURE = 0.07

# API Documentation

## ContrastiveLearningSystem

Main class for text-image contrastive learning.

### Constructor Parameters

- `embedding_dim` (int, optional): Dimension of the shared embedding space. Default: 256
- `tokenizer` (str, optional): Tokenizer type ("mecab" or "sentencepiece"). Default: "mecab"
- `embedding_model` (str, optional): Text embedding model ("bert", "fasttext", or "word2vec"). Default: "bert"
- `use_cuda` (bool, optional): Whether to use GPU acceleration. Default: False

### Methods

#### process_text(text: str) -> List[str]

Tokenize Japanese text into words.

#### process_image(image_path: Union[str, Path]) -> np.ndarray

Extract features from an image.

#### train(text_image_pairs: List[Tuple[str, str]], epochs: int = 10, batch_size: Optional[int] = None) -> None

Train the model on text-image pairs.

#### compute_similarity(text: str, image_path: Union[str, Path]) -> float

Compute similarity between text and image.

#### find_best_matches(query_text: str, image_paths: List[Union[str, Path]], top_k: int = 5) -> List[Tuple[str, float]]

Find the best matching images for a given text query.

#### process_batch(texts: List[str], images: List[Union[str, Path]], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]

Process batches of texts and images efficiently.

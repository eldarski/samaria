# Samaria

Japanese text and image contrastive learning library using CLIP.

## Quick Usage Guide

1. Put your image in the `examples/data/images` folder (for example, `1.jpeg`)

2. Run with Docker (easiest way):

```bash
docker-compose up --build
```

The program will:

- Load your image from `examples/data/images/1.jpeg`
- Analyze the image content
- Show descriptions in Japanese with confidence scores
- Compare the image with sample Japanese text

Example output:

```
Image content: アニメキャラクター(85%)の少女(45%)
Input text: "赤い服の少女"
Similarity score: 0.733
```

Where:

- First line shows what was detected in the image (anime character, girl) with confidence %
- Second line is the text we're comparing with
- Last line shows how well the image matches the text (0-1, higher is better)

## Supported Image Types

- JPEG/JPG
- PNG
- Common web image formats

## Example Japanese Texts

- 赤い服の少女 (Girl in red clothes)
- アニメキャラクター (Anime character)
- 黒髪の少女 (Girl with black hair)

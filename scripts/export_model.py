import torch
from transformers import CLIPModel, CLIPImageProcessor
from transformers.models.clip import CLIPTokenizerFast
import os
from torch import nn


class VisionEncoder(nn.Module):
    """Wrapper for CLIP vision model."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values)
        return outputs.pooler_output


class TextEncoder(nn.Module):
    """Wrapper for CLIP text model with Japanese tokenizer."""

    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids, attention_mask):
        """Accept pre-tokenized inputs."""
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.pooler_output


def export_model():
    # Load model and processors
    model_id = "rinna/japanese-clip-vit-b-16"
    print(f"Loading model {model_id}...")

    # Load model
    model = CLIPModel.from_pretrained(model_id)
    model.eval()

    # Create example inputs
    example_image = torch.randn(1, 3, 224, 224)

    # Create dummy text inputs (batch_size=1, seq_len=77)
    example_input_ids = torch.ones(1, 77, dtype=torch.long)
    example_attention_mask = torch.ones(1, 77, dtype=torch.long)

    # Export vision model
    print("Tracing vision model...")
    vision_wrapper = VisionEncoder(model.vision_model)
    vision_wrapper.eval()
    with torch.no_grad():
        traced_vision = torch.jit.trace(vision_wrapper, example_image, strict=False)

    # Export text model
    print("Tracing text model...")
    text_wrapper = TextEncoder(model.text_model)
    text_wrapper.eval()
    with torch.no_grad():
        traced_text = torch.jit.trace(
            text_wrapper, (example_input_ids, example_attention_mask), strict=False
        )

    # Save models
    os.makedirs("models", exist_ok=True)
    traced_vision.save("models/clip_vision.pt")
    traced_text.save("models/clip_text.pt")
    print("Models exported successfully")


if __name__ == "__main__":
    export_model()

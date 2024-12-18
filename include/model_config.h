#pragma once
#include <string>

namespace samaria {

struct ModelConfig {
    // Vision model settings
    std::string model_id = "rinna/japanese-clip-vit-b-16";
    std::string config_path = "models/config.json";
    std::string model_path = "models/clip_vision.pt";
    std::string text_model_path = "models/clip_text.pt";
    bool use_cuda = false;
    int image_size = 224;
    int patch_size = 16;
    bool use_local_models = false;

    // Text model settings
    int vocab_size = 32000;
    int max_position_embeddings = 512;
    int max_text_length = 512;

    // Embedding settings
    int embedding_dim = 512;
    float logit_scale_init = 2.6592600;
};

}  // namespace samaria
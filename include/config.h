#pragma once

#include <string>

namespace samaria {

struct Config {
    // Model paths
    static constexpr const char* DEFAULT_MODEL_PATH = "models/";
    static constexpr const char* DEFAULT_TOKENIZER_PATH = "models/tokenizer/";

    // Image processing
    static constexpr int DEFAULT_IMAGE_SIZE = 224;
    static constexpr int DEFAULT_BATCH_SIZE = 32;

    // Training
    static constexpr int DEFAULT_EMBEDDING_DIM = 512;
    static constexpr float DEFAULT_LEARNING_RATE = 0.001f;
    static constexpr int DEFAULT_NUM_EPOCHS = 100;

    // Device
    static constexpr const char* DEFAULT_DEVICE = "cuda";
};

}  // namespace samaria
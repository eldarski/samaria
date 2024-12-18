#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model_config.h"

namespace samaria {

class ModelUtils {
public:
    static bool initialize_models(const ModelConfig& config);

    static torch::Tensor preprocess_image(const cv::Mat& image, const ModelConfig& config);

    static std::string get_image_description(const torch::Tensor& features, bool use_cuda = false);

private:
    static torch::jit::Module clip_model_;
    static torch::jit::Module text_model_;
    static const char* MODEL_ID;

    static std::vector<int64_t> tokenize_text(const std::string& text);
};

}  // namespace samaria
#pragma once

#include <torch/torch.h>

#include <memory>
#include <opencv2/core/mat.hpp>

namespace samaria {

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Non-copyable
    ImageProcessor(const ImageProcessor&) = delete;
    ImageProcessor& operator=(const ImageProcessor&) = delete;

    // Process image from file
    torch::Tensor extract_features(const std::string& image_path);

    // Process image from OpenCV Mat
    torch::Tensor extract_features(const cv::Mat& image);

    // Image processing methods
    torch::Tensor extract_features_batch(const std::vector<std::string>& image_paths) {
        std::vector<torch::Tensor> features;
        for (const auto& path : image_paths) {
            features.push_back(extract_features(path));
        }
        return torch::stack(features);
    }
    torch::Tensor extract_features_batch(const std::vector<cv::Mat>& images);

    // Preprocessing
    cv::Mat preprocess_image(const cv::Mat& input_image);

    // ... other methods

    std::string get_description(const std::string& image_path);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace samaria
#pragma once
#include <torch/torch.h>

#include <filesystem>
#include <string>

namespace samaria {
class ModelSerializer {
public:
    static void save_embeddings(const torch::Tensor& embeddings, const std::string& path,
                                const std::string& name);

    static torch::Tensor load_embeddings(const std::string& path, const std::string& name);

    static void save_text_embeddings(const torch::Tensor& embeddings, const std::string& text,
                                     const std::string& base_path);

    static void save_image_embeddings(const torch::Tensor& embeddings,
                                      const std::string& image_path, const std::string& base_path);
};
}  // namespace samaria
#pragma once

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

namespace samaria {

class ContrastiveModel {
public:
    explicit ContrastiveModel(int embedding_dim = 256);
    ~ContrastiveModel();

    // Training methods
    void train(const std::vector<std::pair<std::string, std::string>>& pairs, int epochs = 10);
    void train_batch(const std::vector<std::pair<std::string, std::string>>& pairs);

    // Inference methods
    float compute_similarity(const torch::Tensor& text_emb, const torch::Tensor& img_emb);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace samaria
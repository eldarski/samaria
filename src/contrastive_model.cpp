#include "contrastive_model.h"

#include <iostream>

namespace samaria {

class ContrastiveModel::Impl {
public:
    Impl(int embedding_dim) {}
};

ContrastiveModel::ContrastiveModel(int embedding_dim)
    : pimpl_(std::make_unique<Impl>(embedding_dim)) {}

ContrastiveModel::~ContrastiveModel() = default;

void ContrastiveModel::train(const std::vector<std::pair<std::string, std::string>>& pairs,
                             int epochs) {
    // Stub implementation
    std::cout << "Training on " << pairs.size() << " pairs for " << epochs << " epochs"
              << std::endl;
}

void ContrastiveModel::train_batch(const std::vector<std::pair<std::string, std::string>>& pairs) {
    // Just call train with one epoch
    train(pairs, 1);
}

float ContrastiveModel::compute_similarity(const torch::Tensor& text_emb,
                                           const torch::Tensor& img_emb) {
    // Simple cosine similarity
    auto text_norm = text_emb.norm(2);
    auto img_norm = img_emb.norm(2);
    auto dot_product = torch::sum(text_emb * img_emb);
    return dot_product.item<float>() / (text_norm.item<float>() * img_norm.item<float>());
}

}  // namespace samaria
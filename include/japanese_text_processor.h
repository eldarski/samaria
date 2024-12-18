#pragma once

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

namespace samaria {

enum class TokenizerType { MECAB, SENTENCEPIECE };

enum class EmbeddingModel { BERT, FASTTEXT, WORD2VEC };

class JapaneseTextProcessor {
public:
    JapaneseTextProcessor();
    ~JapaneseTextProcessor();

    // Tokenization methods
    std::vector<std::string> tokenize(const std::string& text);

    std::vector<std::vector<std::string>> tokenize_batch(const std::vector<std::string>& texts,
                                                         int batch_size = 32);

    // Embedding generation
    torch::Tensor generate_embeddings(const std::string& text);

    torch::Tensor generate_embeddings_batch(const std::vector<std::string>& texts,
                                            int batch_size = 32);

    // Token probability distribution
    std::vector<float> compute_token_probabilities(const std::vector<std::string>& tokens);

    // Model save/load
    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace samaria
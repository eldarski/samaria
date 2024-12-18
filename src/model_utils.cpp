#include "model_utils.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>

namespace samaria {

// Static member initialization
torch::jit::Module ModelUtils::clip_model_;
torch::jit::Module ModelUtils::text_model_;
const char* ModelUtils::MODEL_ID = "rinna/japanese-clip-vit-b-16";

bool ModelUtils::initialize_models(const ModelConfig& config) {
    try {
        // Load vision model
        clip_model_ = torch::jit::load(config.model_path);
        // Load text model
        text_model_ = torch::jit::load(config.text_model_path);

        if (config.use_cuda) {
            clip_model_.to(torch::kCUDA);
            text_model_.to(torch::kCUDA);
        }
        clip_model_.eval();
        text_model_.eval();

        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading models: " << e.msg() << std::endl;
        return false;
    }
}

torch::Tensor ModelUtils::preprocess_image(const cv::Mat& image, const ModelConfig& config) {
    cv::Mat processed;
    cv::resize(image, processed, cv::Size(config.image_size, config.image_size));
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    cv::Mat float_mat;
    processed.convertTo(float_mat, CV_32F, 1.0 / 255.0);

    // Normalize using ImageNet stats
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};

    std::vector<cv::Mat> channels(3);
    cv::split(float_mat, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, float_mat);

    auto tensor = torch::from_blob(float_mat.data, {1, config.image_size, config.image_size, 3},
                                   torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}).contiguous();

    // Ensure batch dimension
    if (tensor.dim() == 3) {
        tensor = tensor.unsqueeze(0);
    }

    return tensor.clone();
}

std::vector<int64_t> ModelUtils::tokenize_text(const std::string& text) {
    // Simple character-based tokenization for Japanese
    std::vector<int64_t> tokens = {1};  // BOS token

    // Add character tokens
    for (char c : text) {
        tokens.push_back(static_cast<int64_t>(c));
    }

    // Add EOS token and pad to max length
    tokens.push_back(2);  // EOS token
    while (tokens.size() < 77) {
        tokens.push_back(0);  // PAD token
    }

    return tokens;
}

std::string ModelUtils::get_image_description(const torch::Tensor& image_features, bool use_cuda) {
    torch::NoGradGuard no_grad;

    // List of Japanese image categories to check against
    const std::vector<std::pair<std::string, std::string>> categories = {
        {"アニメキャラクター", "anime character"},
        {"少女", "girl"},
        {"女性", "woman"},
        {"男性", "man"},
        {"戦士", "warrior"},
        {"魔法使い", "mage"},
        {"赤い服", "red clothes"},
        {"黒髪", "black hair"},
        {"金髪", "blonde hair"},
        {"青い目", "blue eyes"},
        {"制服", "uniform"},
        {"鎧", "armor"},
        {"戦闘的", "combat"},
        {"笑顔", "smiling"},
        {"真剣な", "serious"},
        {"悲しい", "sad"},
        {"怒り", "angry"},
        {"炎", "fire"},
        {"水", "water"},
        {"雷", "lightning"},
        {"魔法", "magic"},
        {"光", "light"},
        {"闇", "darkness"},
        {"風景", "scenery"},
        {"森", "forest"},
        {"海", "ocean"},
        {"空", "sky"},
        {"夜空", "night sky"},
        {"花", "flower"},
        {"街", "city"},
        {"学校", "school"},
        {"戦場", "battlefield"},
        {"猫", "cat"},
        {"剣", "sword"},
        {"本", "book"},
        {"武器", "weapon"}};

    // Get image embeddings from our wrapped model
    auto image_embedding = clip_model_.forward({image_features}).toTensor();

    // Ensure correct shape [1, 512]
    if (image_embedding.dim() == 1) {
        image_embedding = image_embedding.unsqueeze(0);
    }

    // Get text embeddings for all categories
    std::vector<torch::Tensor> text_embeddings;
    for (const auto& category : categories) {
        // Tokenize text
        auto tokens = tokenize_text(category.second);
        auto attention_mask = std::vector<int64_t>(77, 1);

        // Convert to tensors
        auto input_ids = torch::tensor({tokens}, torch::kLong);
        auto mask = torch::tensor({attention_mask}, torch::kLong);

        // Get text features
        auto text_inputs = text_model_.forward({input_ids, mask}).toTensor();
        text_embeddings.push_back(text_inputs);
    }

    // Stack all text embeddings
    auto text_features = torch::stack(text_embeddings);

    // Calculate similarities
    auto similarities = torch::matmul(image_embedding, text_features.transpose(0, 1));
    auto probs =
        torch::softmax(similarities * 75.0, 1);  // Lower temperature for more balanced predictions

    // Get top categories with confidence threshold
    auto values_indices = torch::topk(probs, 3);  // Get top 3 predictions
    auto top_indices = std::get<1>(values_indices);
    auto top_probs = std::get<0>(values_indices);

    // Build description using top categories
    std::stringstream ss;
    ss << categories[top_indices[0][0].item<int64_t>()].first;
    float primary_conf = top_probs[0][0].item<float>();
    float secondary_conf = top_probs[0][1].item<float>();
    float tertiary_conf = top_probs[0][2].item<float>();

    // Add confidence to primary category
    ss << "(" << static_cast<int>(primary_conf * 100) << "%)";

    // Add secondary category if confidence is high enough
    if (secondary_conf > 0.15) {  // Lower threshold for more descriptions
        ss << "の" << categories[top_indices[0][1].item<int64_t>()].first;
        ss << "(" << static_cast<int>(secondary_conf * 100) << "%)";

        // Add tertiary category for more detail if confidence is good
        if (tertiary_conf > 0.1 && (primary_conf - tertiary_conf) < 0.4) {  // Check confidence gap
            ss << "と" << categories[top_indices[0][2].item<int64_t>()].first;
            ss << "(" << static_cast<int>(tertiary_conf * 100) << "%)";
        }
    }

    return ss.str();
}

}  // namespace samaria
#include "model_serialization.h"
#include <torch/torch.h>
#include <filesystem>
#include <stdexcept>

namespace jtext
{
    void ModelSerializer::save_model(
        const std::string &path,
        const torch::nn::Module &model,
        const torch::optim::Optimizer &optimizer)
    {
        try
        {
            // Create directory if it doesn't exist
            std::filesystem::create_directories(
                std::filesystem::path(path).parent_path());

            // Save model state
            torch::save(model.state_dict(), path + ".model");

            // Save optimizer state
            torch::save(optimizer.state_dict(), path + ".optimizer");
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to save model: " + std::string(e.what()));
        }
    }

    void ModelSerializer::load_model(
        const std::string &path,
        torch::nn::Module &model,
        torch::optim::Optimizer &optimizer)
    {
        try
        {
            // Check if files exist
            if (!std::filesystem::exists(path + ".model") ||
                !std::filesystem::exists(path + ".optimizer"))
            {
                throw std::runtime_error("Model files not found at: " + path);
            }

            // Load model state
            torch::load(model.state_dict(), path + ".model");

            // Load optimizer state
            torch::load(optimizer.state_dict(), path + ".optimizer");
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Failed to load model: " + std::string(e.what()));
        }
    }
}
#pragma once
#include <stdexcept>
#include <string>

namespace jtext
{
    class ModelError : public std::runtime_error
    {
    public:
        explicit ModelError(const std::string &message) : std::runtime_error(message) {}
    };

    class TokenizerError : public std::runtime_error
    {
    public:
        explicit TokenizerError(const std::string &message) : std::runtime_error(message) {}
    };

    class ConfigurationError : public std::runtime_error
    {
    public:
        explicit ConfigurationError(const std::string &message) : std::runtime_error(message) {}
    };
} // namespace jtext
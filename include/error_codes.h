#pragma once
#include <string>

namespace samaria {
enum class ErrorCode {
    SUCCESS = 0,
    MODEL_LOAD_ERROR = 1,
    TOKENIZATION_ERROR = 2,
    CUDA_ERROR = 3,
    FILE_IO_ERROR = 4,
    INVALID_INPUT = 5,
    UNKNOWN_ERROR = 999
};

std::string error_code_to_string(ErrorCode code);
}  // namespace samaria
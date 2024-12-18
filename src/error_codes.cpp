#include "error_codes.h"

namespace jtext
{
    std::string error_code_to_string(ErrorCode code)
    {
        switch (code)
        {
        case ErrorCode::SUCCESS:
            return "Success";
        case ErrorCode::MODEL_LOAD_ERROR:
            return "Failed to load model";
        case ErrorCode::TOKENIZER_ERROR:
            return "Tokenizer error";
        case ErrorCode::CUDA_ERROR:
            return "CUDA error";
        case ErrorCode::FILE_IO_ERROR:
            return "File I/O error";
        case ErrorCode::INVALID_INPUT:
            return "Invalid input";
        default:
            return "Unknown error";
        }
    }
}
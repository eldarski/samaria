#include "logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace samaria {
std::shared_ptr<spdlog::logger> Logger::logger_;

void Logger::init(const std::string& log_level) {
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt("samaria");
        logger_->set_level(spdlog::level::from_str(log_level));
    }
}

std::shared_ptr<spdlog::logger> Logger::get() {
    if (!logger_) {
        init();
    }
    return logger_;
}
}  // namespace samaria
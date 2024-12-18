#pragma once
#include <spdlog/spdlog.h>

#include <string>

namespace samaria {
class Logger {
public:
    static void init(const std::string& log_level = "info");
    static std::shared_ptr<spdlog::logger> get();

private:
    static std::shared_ptr<spdlog::logger> logger_;
};
}  // namespace samaria
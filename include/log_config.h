#pragma once

#include <string>

namespace samaria {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class LogConfig {
public:
    static void setLogLevel(LogLevel level);
    static void setLogFile(const std::string& path);
    static LogLevel getLogLevel();
    static bool isDebugEnabled();

private:
    static LogLevel current_level_;
    static std::string log_file_;
};

}  // namespace samaria
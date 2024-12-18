find_path(SPDLOG_INCLUDE_DIR
    NAMES spdlog/spdlog.h
    PATHS
    /opt/homebrew/include
    NO_DEFAULT_PATH
)

find_library(SPDLOG_LIBRARY
    NAMES spdlog
    PATHS
    /opt/homebrew/lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spdlog DEFAULT_MSG
    SPDLOG_LIBRARY SPDLOG_INCLUDE_DIR)

if(spdlog_FOUND AND NOT TARGET spdlog::spdlog)
    add_library(spdlog::spdlog UNKNOWN IMPORTED)
    set_target_properties(spdlog::spdlog PROPERTIES
        IMPORTED_LOCATION "${SPDLOG_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SPDLOG_INCLUDE_DIR}"
    )
endif() 
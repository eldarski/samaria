find_path(MECAB_INCLUDE_DIR
    NAMES mecab.h
    PATHS
    /usr/include
    /usr/local/include
    /opt/homebrew/include
)

find_library(MECAB_LIBRARY
    NAMES mecab
    PATHS
    /usr/lib
    /usr/local/lib
    /opt/homebrew/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MeCab DEFAULT_MSG
    MECAB_LIBRARY MECAB_INCLUDE_DIR) 
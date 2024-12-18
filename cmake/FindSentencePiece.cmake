find_path(SENTENCEPIECE_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    PATHS
    /opt/homebrew/include
    NO_DEFAULT_PATH
)

find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece
    PATHS
    /opt/homebrew/lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SentencePiece DEFAULT_MSG
    SENTENCEPIECE_LIBRARY SENTENCEPIECE_INCLUDE_DIR)

if(SentencePiece_FOUND AND NOT TARGET SentencePiece::SentencePiece)
    add_library(SentencePiece::SentencePiece UNKNOWN IMPORTED)
    set_target_properties(SentencePiece::SentencePiece PROPERTIES
        IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
    )
endif() 
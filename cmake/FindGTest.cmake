find_path(GTEST_INCLUDE_DIR
    NAMES gtest/gtest.h
    PATHS
    /opt/homebrew/include
    NO_DEFAULT_PATH
)

find_library(GTEST_LIBRARY
    NAMES gtest
    PATHS
    /opt/homebrew/lib
    NO_DEFAULT_PATH
)

find_library(GTEST_MAIN_LIBRARY
    NAMES gtest_main
    PATHS
    /opt/homebrew/lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTest DEFAULT_MSG
    GTEST_LIBRARY GTEST_MAIN_LIBRARY GTEST_INCLUDE_DIR)

if(GTest_FOUND AND NOT TARGET GTest::GTest)
    add_library(GTest::GTest UNKNOWN IMPORTED)
    set_target_properties(GTest::GTest PROPERTIES
        IMPORTED_LOCATION "${GTEST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
    )
endif()

if(GTest_FOUND AND NOT TARGET GTest::Main)
    add_library(GTest::Main UNKNOWN IMPORTED)
    set_target_properties(GTest::Main PROPERTIES
        IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
    )
endif() 
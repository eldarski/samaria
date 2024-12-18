cmake_minimum_required(VERSION 3.15)

# Enable testing
option(BUILD_TESTING "Build tests" OFF)

# Force arm64 architecture for Apple Silicon
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
set(CMAKE_APPLE_SILICON_PROCESSOR "arm64")

project(samaria_contrast VERSION 0.1.0 LANGUAGES CXX)

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get git revision
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(GIT_REVISION "unknown")
endif()

# Get release date
string(TIMESTAMP RELEASE_DATE "%Y-%m-%d")

# Configure Python package version
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/samaria_contrast/version.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/python/samaria_contrast/version.py
    @ONLY
)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(CUDA QUIET)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(CURL REQUIRED)

# Set CUDA flags if available
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DWITH_CUDA)
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(bindings)
add_subdirectory(python)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Installation configuration
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.h"
)

# Install library
install(
    TARGETS samaria_contrast
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Generate and install package config
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Generate and install package version
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# Configure version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h
    @ONLY
)

target_link_libraries(samaria_contrast PRIVATE 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    CURL::libcurl
) 
add_library(samaria_contrast
    image_processor.cpp
    model_config.cpp
    model_utils.cpp
)

target_include_directories(samaria_contrast
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(samaria_contrast
    PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    spdlog::spdlog
) 
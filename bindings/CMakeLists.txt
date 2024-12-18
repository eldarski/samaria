pybind11_add_module(samaria_bindings
    python_bindings.cpp
)

target_link_libraries(samaria_bindings
    PRIVATE
    samaria_contrast
    pybind11::module
)

set_target_properties(samaria_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/samaria_contrast
    LIBRARY_OUTPUT_NAME "_samaria_bindings"
) 
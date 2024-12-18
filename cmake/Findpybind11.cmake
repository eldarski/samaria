find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED) 
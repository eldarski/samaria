find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(Torch_DIR "${TORCH_INSTALL_PREFIX}/Torch")
find_package(Torch CONFIG REQUIRED) 
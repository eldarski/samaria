# Python package configuration
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    @ONLY
)

# Install Python package
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/samaria_contrast
    DESTINATION ${Python_SITEARCH}
    FILES_MATCHING PATTERN "*.py"
)

# Add Python tests to CTest
add_test(
    NAME python_tests
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
) 
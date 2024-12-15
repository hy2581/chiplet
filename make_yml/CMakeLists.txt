cmake_minimum_required(VERSION 3.12) # or higher, depending on your needs

# Project name
project(GenerateYAML LANGUAGES CXX)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add executable
add_executable(generate_exact_yaml test.cpp)

# Compiler flags
if (MSVC)
    # MSVC uses /W3 as a default warning level
    target_compile_options(generate_exact_yaml PRIVATE /W4)
else()
    # GCC or Clang
    target_compile_options(generate_exact_yaml PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Optionally, add any libraries or paths if needed
# For example:
# find_package(Threads REQUIRED)
# target_link_libraries(generate_exact_yaml Threads::Threads)

# Optionally, specify an installation path for the executable
# install(TARGETS generate_exact_yaml DESTINATION bin)

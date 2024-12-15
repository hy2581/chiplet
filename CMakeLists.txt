cmake_minimum_required(VERSION 3.10)
project(MyExecutable)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 ROOT_DIR 变量，并导出为环境变量
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENV{MYPROJECT_ROOT_DIR} ${ROOT_DIR})

# 打印重要信息
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "ROOT_DIR: ${ROOT_DIR}")

# 检查必要的源文件和目录是否存在
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/make_yml/test.cpp")
    message(FATAL_ERROR "Source file 'make_yml/test.cpp' not found in ${PROJECT_SOURCE_DIR}")
endif()
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/include/gem5_class.hpp")
    message(FATAL_ERROR "Header file 'include/gem5_class.hpp' not found in ${PROJECT_SOURCE_DIR}")
endif()

# 设置 include 和 source 路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# 查找 nlohmann_json 包
find_package(nlohmann_json CONFIG REQUIRED)

# 查找 Boost 包
find_package(Boost REQUIRED COMPONENTS filesystem)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found")
endif()

# 添加可执行文件
add_executable(MyExecutable make_yml/test.cpp ${SOURCES})
target_compile_definitions(MyExecutable PRIVATE ROOT_DIR="${ROOT_DIR}")

# 链接 nlohmann_json 库和 Boost filesystem 库
target_link_libraries(MyExecutable PRIVATE nlohmann_json::nlohmann_json Boost::filesystem)

set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)
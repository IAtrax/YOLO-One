cmake_minimum_required(VERSION 3.15)
project(YOLOOne VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(cpp/include)

# Add subdirectories
add_subdirectory(cpp)
add_subdirectory(bindings)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(USE_CUDA "Enable CUDA support" OFF)

if(BUILD_TESTS)
    add_subdirectory(cpp/tests)
endif()

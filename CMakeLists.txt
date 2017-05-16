project(MultiVariatePointProcess C CXX)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE Release)

file(GLOB cpp_files src/*.cc)

include_directories(include 3rd-party)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -Wall")

add_library(ptpack ${cpp_files})

add_subdirectory(example)
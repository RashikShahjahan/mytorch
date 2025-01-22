cmake_minimum_required(VERSION 3.10)
project(mytorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(xtensor REQUIRED)

add_executable(mytorch tensor.cc)
target_link_libraries(mytorch xtensor) 